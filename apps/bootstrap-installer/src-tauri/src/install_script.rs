//! Resolves and downloads `scripts/install.ps1` (and `install.sh`).
//!
//! Resolution order:
//!   1. Dev shortcut: a sibling repo checkout via $HERMES_SETUP_DEV_REPO_ROOT
//!      env var. Lets devs iterate without re-publishing the script.
//!   2. Bundled fallback: if the installer was bundled with a script (e.g.
//!      tauri's `resource` mechanism), serve from there. Not used today.
//!   3. Network: download from GitHub raw at a pinned commit or branch.
//!      Commit pins are immutable; branch pins are HEAD-tracking.
//!
//! Mirrors `apps/desktop/electron/bootstrap-runner.ts`'s `resolveInstallScript`,
//! but the dev-checkout resolution is driven by an env var rather than the
//! Electron app's APP_ROOT/../.. trick, because Hermes-Setup.exe is meant
//! to live OUTSIDE any repo checkout.

use anyhow::{anyhow, Context, Result};
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::paths;

/// Identity of the install.ps1 we'll execute. Used by both the manifest
/// fetch and the per-stage runs.
#[derive(Debug, Clone)]
pub struct ResolvedScript {
    pub path: PathBuf,
    pub source: ScriptSource,
    /// Commit pin (40-char SHA) if known. install.ps1's `-Commit` arg is
    /// what makes the repo stage clone the exact tested SHA.
    pub commit: Option<String>,
    pub branch: Option<String>,
    // All clones retain this run's executable until the last user is done.
    // Dev and bundled sources belong to their checkout/package, not this run.
    _cache_owner: Option<Arc<CachedScript>>,
}

#[derive(Debug)]
struct CachedScript {
    directory: PathBuf,
}

impl CachedScript {
    fn reserve(cache_root: &Path) -> Result<Arc<Self>> {
        std::fs::create_dir_all(cache_root).with_context(|| {
            format!("creating bootstrap-cache parent dir {}", cache_root.display())
        })?;
        let directory = cache_root.join(format!("run-{}", uuid::Uuid::new_v4()));
        // Reserve before acquiring ownership: an existing directory must never
        // become ours to delete, even in the unlikely event of a name collision.
        std::fs::create_dir(&directory)
            .with_context(|| format!("reserving script directory {}", directory.display()))?;
        Ok(Arc::new(Self { directory }))
    }
}

impl Drop for CachedScript {
    fn drop(&mut self) {
        match std::fs::remove_dir_all(&self.directory) {
            Ok(()) => {},
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {},
            Err(err) => tracing::warn!("could not remove script directory {}: {err}", self.directory.display()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptSource {
    DevCheckout,
    Bundled,
    Downloaded,
}

/// What flavor of script (Windows .ps1 vs Unix .sh).
#[derive(Debug, Clone, Copy)]
pub enum ScriptKind {
    Ps1,
    Sh,
}

impl ScriptKind {
    pub fn for_current_os() -> Self {
        if cfg!(target_os = "windows") {
            Self::Ps1
        } else {
            Self::Sh
        }
    }

    fn filename(&self) -> &'static str {
        match self {
            Self::Ps1 => "install.ps1",
            Self::Sh => "install.sh",
        }
    }
}

/// Validates a string looks like a git SHA (7+ hex chars). Mirrors
/// `STAMP_COMMIT_RE` from bootstrap-runner.ts.
fn is_valid_commit(s: &str) -> bool {
    let len = s.len();
    (7..=40).contains(&len) && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Resolves the install script to use for this run.
///
/// `pin` is the commit-or-branch from either Hermes-Setup's build-time
/// constant (compiled into the installer) or a runtime override.
pub async fn resolve(
    kind: ScriptKind,
    pin: &Pin,
    emit_log: &impl Fn(&str),
    cancel_rx: &mut Option<mpsc::Receiver<()>>,
) -> Result<ResolvedScript> {
    check_cancelled(cancel_rx)?;
    // 1. Dev shortcut.
    if let Ok(repo_root) = std::env::var("HERMES_SETUP_DEV_REPO_ROOT") {
        let candidate = PathBuf::from(repo_root).join("scripts").join(kind.filename());
        if candidate.exists() {
            emit_log(&format!(
                "[bootstrap] dev mode — using local {} at {}",
                kind.filename(),
                candidate.display()
            ));
            return Ok(ResolvedScript {
                path: candidate,
                source: ScriptSource::DevCheckout,
                commit: pin.commit.clone(),
                branch: pin.branch.clone(),
                _cache_owner: None,
            });
        }
    }

    // 2. (Not implemented) bundled fallback.

    // 3. Network. Pin must be a real commit or a branch ref.
    //
    // Always download; a previously downloaded script is never reused. A
    // stale script drives a tree it predates (the repository stage follows
    // the live branch), and a failed download is fatal so Retry refetches.
    let commit_or_ref = match (&pin.commit, &pin.branch) {
        (Some(c), _) if is_valid_commit(c) => c.clone(),
        (_, Some(b)) if !b.trim().is_empty() => b.clone(),
        (Some(other), _) => {
            return Err(anyhow!(
                "install script pin commit `{other}` is not a valid git SHA"
            ));
        }
        _ => {
            return Err(anyhow!(
                "no install-script pin supplied — installer cannot resolve a script source"
            ));
        }
    };

    emit_log(&format!(
        "[bootstrap] downloading {} for {} from GitHub",
        kind.filename(),
        truncate_ref(&commit_or_ref)
    ));
    let script = download_resolved_script(
        kind, pin, &paths::bootstrap_cache_dir(), &download_client()?,
        &script_urls(kind, &commit_or_ref), cancel_rx,
    ).await?;
    emit_log(&format!("[bootstrap] downloaded to {}", script.path.display()));
    Ok(script)
}

#[derive(Debug, Clone, Default)]
pub struct Pin {
    pub commit: Option<String>,
    pub branch: Option<String>,
}

async fn download_resolved_script(
    kind: ScriptKind,
    pin: &Pin,
    cache_root: &Path,
    client: &reqwest::Client,
    urls: &[String],
    cancel_rx: &mut Option<mpsc::Receiver<()>>,
) -> Result<ResolvedScript> {
    check_cancelled(cancel_rx)?;
    let owner = CachedScript::reserve(cache_root)?;
    let dest = owner.directory.join(kind.filename());
    download_from_urls_cancellable(kind, &dest, client, urls, cancel_rx).await?;
    Ok(ResolvedScript {
        path: dest,
        source: ScriptSource::Downloaded,
        commit: pin.commit.clone(),
        branch: pin.branch.clone(),
        _cache_owner: Some(owner),
    })
}

fn truncate_ref(s: &str) -> &str {
    if is_valid_commit(s) && s.len() >= 12 {
        &s[..12]
    } else {
        s
    }
}

/// UTF-8 BOM. Windows PowerShell 5.1 reads a BOM-less `.ps1` using the system
/// ANSI code page; a leading BOM is what tells it the file is UTF-8. The
/// `irm | iex` / `[scriptblock]::Create` path strips BOMs on purpose, but the
/// GUI bootstrap runs the *cached file* via `-File`, so we write the opposite
/// (#67193).
const UTF8_BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// Prepare bytes for the on-disk bootstrap cache.
///
/// `.ps1` files get a UTF-8 BOM (unless one is already present). `.sh` files
/// are left unchanged — a BOM would break `#!/usr/bin/env bash`.
pub(crate) fn prepare_cached_script_bytes(kind: ScriptKind, bytes: &[u8]) -> Vec<u8> {
    match kind {
        ScriptKind::Ps1 => {
            if bytes.starts_with(UTF8_BOM) {
                bytes.to_vec()
            } else {
                let mut out = Vec::with_capacity(UTF8_BOM.len() + bytes.len());
                out.extend_from_slice(UTF8_BOM);
                out.extend_from_slice(bytes);
                out
            }
        }
        ScriptKind::Sh => bytes.to_vec(),
    }
}

/// Only `main` has an equivalent project-site endpoint. A commit or another
/// branch must never silently execute the live-main installer instead.
fn script_urls(kind: ScriptKind, commit_or_ref: &str) -> Vec<String> {
    let mut raw = reqwest::Url::parse("https://raw.githubusercontent.com/")
        .expect("constant raw URL");
    raw.path_segments_mut()
        .expect("raw URL has path segments")
        .extend(["NousResearch", "hermes-agent"])
        .extend(commit_or_ref.split('/'))
        .extend(["scripts", kind.filename()]);
    let mut urls = vec![raw.to_string()];
    if commit_or_ref == "main" {
        urls.push(format!(
            "https://hermes-agent.nousresearch.com/{}",
            kind.filename()
        ));
    }
    urls
}

fn fallback_allowed_status(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 403 | 429) || status.is_server_error()
}

const MAX_SCRIPT_BYTES: usize = 2 * 1024 * 1024;

/// Reject error pages and corrupt text before publishing an executable file.
/// This is a transport sanity check, not a substitute for parsing a script.
fn validate_script_body(bytes: &[u8], content_type: &str) -> Result<()> {
    if bytes.is_empty() {
        return Err(anyhow!("empty install script body"));
    }
    if bytes.contains(&0) {
        return Err(anyhow!("install script body contains NUL bytes"));
    }
    let text = std::str::from_utf8(bytes).context("install script is not UTF-8")?;
    let content_type = content_type.to_ascii_lowercase();
    let prefix = text.trim_start_matches('\u{feff}').trim_start().to_ascii_lowercase();
    if content_type.contains("html")
        || content_type.contains("json")
        || prefix.starts_with("<!doctype html")
        || prefix.starts_with("<html")
    {
        return Err(anyhow!("install script response is an error document"));
    }
    if text.trim_start_matches('\u{feff}').trim().is_empty() {
        return Err(anyhow!("empty install script body"));
    }
    Ok(())
}

// Failed local writes and renames must not leave a partial executable behind.
// Each download owns one name; overlapping Retry calls never share a temp file.
struct TempScript(PathBuf);

impl Drop for TempScript {
    fn drop(&mut self) {
        match std::fs::remove_file(&self.0) {
            Ok(()) => {},
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {},
            Err(err) => tracing::warn!("could not remove temporary script {}: {err}", self.0.display()),
        }
    }
}

/// Download a complete response before touching the destination. Network and
/// body failures can use the next equivalent source; local I/O failures cannot.
fn download_client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .https_only(true)
        .redirect(reqwest::redirect::Policy::limited(5))
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .context("building download client")
}

pub(crate) fn check_cancelled(cancel_rx: &mut Option<mpsc::Receiver<()>>) -> Result<()> {
    if cancel_rx.as_mut().is_some_and(|rx| rx.try_recv().is_ok()) {
        return Err(anyhow!("bootstrap cancelled by user"));
    }
    Ok(())
}

/// Only network futures are dropped on Cancel. The bounded local publication
/// has no filesystem futures that could outlive their owning cleanup guards.
async fn await_network<T>(operation: impl Future<Output = T>, cancel_rx: &mut Option<mpsc::Receiver<()>>) -> Result<T> {
    check_cancelled(cancel_rx)?;
    tokio::pin!(operation);
    match cancel_rx.as_mut() {
        Some(rx) => tokio::select! {
            biased;
            signal = rx.recv() => {
                if signal.is_some() {
                    Err(anyhow!("bootstrap cancelled by user"))
                } else {
                    Ok(operation.await)
                }
            },
            result = &mut operation => Ok(result),
        },
        None => Ok(operation.await),
    }
}

// Internal URL/client seam keeps network tests on loopback without weakening
// the HTTPS-only production client or depending on a public service.
#[cfg(test)]
async fn download_from_urls(
    kind: ScriptKind,
    dest_path: &Path,
    client: &reqwest::Client,
    urls: &[String],
) -> Result<()> {
    download_from_urls_cancellable(kind, dest_path, client, urls, &mut None).await
}

async fn download_from_urls_cancellable(
    kind: ScriptKind,
    dest_path: &Path,
    client: &reqwest::Client,
    urls: &[String],
    cancel_rx: &mut Option<mpsc::Receiver<()>>,
) -> Result<()> {
    let mut failures = Vec::new();
    for (index, url) in urls.iter().enumerate() {
        let mut response = match await_network(client
            .get(url)
            .header("User-Agent", "hermes-setup/0.0.1")
            .send(), cancel_rx).await?
        {
            Ok(response) => response,
            Err(err) => {
                failures.push(format!("GET {url}: {err}"));
                continue;
            }
        };
        if response.status() != reqwest::StatusCode::OK {
            let status = response.status();
            failures.push(format!(
                "Failed to download {}: HTTP {} from {}",
                kind.filename(), status, url
            ));
            if !fallback_allowed_status(status) {
                break;
            }
            continue;
        }

        let content_type = response.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_owned();
        let body = await_network(async {
            if response.content_length().is_some_and(|length| length > MAX_SCRIPT_BYTES as u64) {
                return Err(anyhow!("install script exceeds {MAX_SCRIPT_BYTES} bytes"));
            }
            let mut bytes = Vec::new();
            while let Some(chunk) = response.chunk().await.context("reading install script body")? {
                if chunk.len() > MAX_SCRIPT_BYTES - bytes.len() {
                    return Err(anyhow!("install script exceeds {MAX_SCRIPT_BYTES} bytes"));
                }
                bytes.extend_from_slice(&chunk);
            }
            validate_script_body(&bytes, &content_type)?;
            Ok::<_, anyhow::Error>(bytes)
        }, cancel_rx).await?;
        let bytes = match body {
            Ok(bytes) => prepare_cached_script_bytes(kind, &bytes),
            Err(err) => {
                failures.push(format!("body of {url}: {err:#}"));
                continue;
            }
        };
        check_cancelled(cancel_rx)?;

        if let Some(parent) = dest_path.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("creating bootstrap-cache parent dir {}", parent.display())
            })?;
        }
        let temp = TempScript(dest_path.with_extension(format!(
            "{}.{}.tmp",
            dest_path.extension().and_then(|s| s.to_str()).unwrap_or("script"),
            uuid::Uuid::new_v4()
        )));
        // Publication is bounded to 2 MiB and runs to completion in this
        // scope. Dropping the resolver can never detach a filesystem worker
        // that recreates or renames a file after its owner was cleaned up.
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp.0)
            .with_context(|| format!("creating temp file {}", temp.0.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("writing temp file {}", temp.0.display()))?;
        file.flush().context("flushing temp file")?;
        drop(file);
        check_cancelled(cancel_rx)?;
        std::fs::rename(&temp.0, dest_path).with_context(|| {
            format!("renaming {} → {}", temp.0.display(), dest_path.display())
        })?;
        check_cancelled(cancel_rx)?;
        if index > 0 {
            tracing::info!("install script served by fallback rung {url}");
        }
        return Ok(());
    }
    Err(anyhow!("{}", failures.join("\n")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn script_ladder_prefers_raw_then_the_site() {
        let urls = script_urls(ScriptKind::Ps1, "main");
        assert_eq!(urls.len(), 2);
        assert!(urls[0].starts_with("https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.ps1"));
        assert_eq!(urls[1], "https://hermes-agent.nousresearch.com/install.ps1");
        let sh = script_urls(ScriptKind::Sh, "main");
        assert_eq!(sh[1], "https://hermes-agent.nousresearch.com/install.sh");
        for pin in ["abc123456", "release/1.2.3", "refs/heads/main", "Main"] {
            assert_eq!(script_urls(ScriptKind::Sh, pin).len(), 1, "{pin}");
        }
    }

    #[test]
    fn fallback_rungs_cover_edges_and_hiccups_but_not_wrong_refs() {
        // Availability failures move to the next rung …
        assert!(fallback_allowed_status(reqwest::StatusCode::FORBIDDEN));
        assert!(fallback_allowed_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(fallback_allowed_status(reqwest::StatusCode::BAD_GATEWAY));
        assert!(fallback_allowed_status(reqwest::StatusCode::SERVICE_UNAVAILABLE));
        assert!(fallback_allowed_status(reqwest::StatusCode::GATEWAY_TIMEOUT));
        assert!(fallback_allowed_status(reqwest::StatusCode::INTERNAL_SERVER_ERROR));
        // … a definitive 4xx means the ref itself is wrong; stop the ladder.
        assert!(!fallback_allowed_status(reqwest::StatusCode::NOT_FOUND));
        assert!(!fallback_allowed_status(reqwest::StatusCode::GONE));
        assert!(!fallback_allowed_status(reqwest::StatusCode::UNAUTHORIZED));
        assert!(!fallback_allowed_status(reqwest::StatusCode::OK));
    }

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("hermes-script-test-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    struct TestServer {
        address: std::net::SocketAddr,
        requests: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    impl TestServer {
        fn new(responses: Vec<Vec<u8>>) -> Self {
            Self::start(responses, false)
        }

        fn start(responses: Vec<Vec<u8>>, hold_connection: bool) -> Self {
            use std::io::{Read, Write};
            use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
            use std::sync::Arc;
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let address = listener.local_addr().unwrap();
            listener.set_nonblocking(true).unwrap();
            let requests = Arc::new(AtomicUsize::new(0));
            let stop = Arc::new(AtomicBool::new(false));
            let seen = requests.clone();
            let stopped = stop.clone();
            let thread = std::thread::spawn(move || {
                while !stopped.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((mut stream, _)) => {
                            stream.set_read_timeout(Some(std::time::Duration::from_secs(2))).unwrap();
                            stream.set_write_timeout(Some(std::time::Duration::from_secs(2))).unwrap();
                            let mut request = Vec::new();
                            let mut buffer = [0; 1024];
                            while !request.windows(4).any(|end| end == b"\r\n\r\n") {
                                match stream.read(&mut buffer) {
                                    Ok(0) | Err(_) => break,
                                    Ok(size) => request.extend_from_slice(&buffer[..size]),
                                }
                            }
                            let index = seen.fetch_add(1, Ordering::SeqCst);
                            if let Some(response) = responses.get(index) {
                                let _ = stream.write_all(response);
                            }
                            if hold_connection {
                                while !stopped.load(Ordering::SeqCst) {
                                    std::thread::sleep(std::time::Duration::from_millis(5));
                                }
                            }
                        },
                        Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(std::time::Duration::from_millis(5));
                        },
                        Err(err) => panic!("accepting test connection: {err}"),
                    }
                }
            });
            Self { address, requests, stop, thread: Some(thread) }
        }

        fn urls(&self) -> Vec<String> {
            vec![format!("http://{}/raw", self.address), format!("http://{}/site", self.address)]
        }

        fn count(&self) -> usize {
            self.requests.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
            self.thread.take().unwrap().join().unwrap();
        }
    }

    fn response(status: u16, content_type: &str, body: &[u8]) -> Vec<u8> {
        let mut response = format!(
            "HTTP/1.1 {status} test\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        ).into_bytes();
        response.extend_from_slice(body);
        response
    }

    fn test_client() -> reqwest::Client {
        reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::limited(5))
            .timeout(std::time::Duration::from_secs(3))
            .build().unwrap()
    }

    fn assert_only_destination(dir: &TestDir, dest: &Path) {
        let entries: Vec<_> = std::fs::read_dir(&dir.0).unwrap()
            .map(|entry| entry.unwrap().path()).collect();
        assert_eq!(entries, vec![dest.to_path_buf()], "temporary scripts must be cleaned up");
    }

    #[tokio::test]
    async fn resolved_runs_keep_independent_executables_and_clone_owned_lifetimes() {
        for (first_ref, second_ref) in [("feature/a", "feature_a"), ("main", "main")] {
            for kind in [ScriptKind::Sh, ScriptKind::Ps1] {
                let dir = TestDir::new();
                let first_body = b"echo first run";
                let second_body = b"echo second run";
                let server = TestServer::new(vec![
                    response(200, "text/plain", first_body),
                    response(200, "text/plain", second_body),
                ]);
                let client = test_client();
                let urls = server.urls();
                let first = download_resolved_script(kind,
                    &Pin { commit: None, branch: Some(first_ref.into()) },
                    &dir.0, &client, &urls, &mut None).await.unwrap();
                let first_path = first.path.clone();
                let retained = first.clone();
                let second = download_resolved_script(kind,
                    &Pin { commit: None, branch: Some(second_ref.into()) },
                    &dir.0, &client, &urls, &mut None).await.unwrap();
                let second_path = second.path.clone();
                assert_ne!(first_path, second_path, "runs sharing a cache must never share an executable");
                assert_eq!(std::fs::read(&first_path).unwrap(), prepare_cached_script_bytes(kind, first_body));
                assert_eq!(std::fs::read(&second_path).unwrap(), prepare_cached_script_bytes(kind, second_body));
                assert_eq!(retained.branch.as_deref(), Some(first_ref));
                assert_eq!(second.branch.as_deref(), Some(second_ref));
                assert_eq!(server.count(), 2);
                drop(first);
                assert!(first_path.exists(), "a retained clone still executes this script");
                drop(second);
                assert!(!second_path.exists());
                assert!(first_path.exists(), "cleaning one run must not delete another");
                drop(retained);
                assert!(!first_path.exists());
                assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
            }
        }
    }

    #[tokio::test]
    async fn failed_resolver_releases_only_its_own_directory() {
        let dir = TestDir::new();
        let server = TestServer::new(vec![
            response(200, "text/plain", b"echo retained"),
            response(404, "text/plain", b"missing"),
        ]);
        let pin = Pin { commit: Some("abc123456".into()), branch: None };
        let client = test_client();
        let urls = server.urls();
        let retained = download_resolved_script(ScriptKind::Sh, &pin,
            &dir.0, &client, &urls, &mut None).await.unwrap();
        assert!(download_resolved_script(ScriptKind::Sh, &pin,
            &dir.0, &client, &urls, &mut None).await.is_err());
        assert_eq!(std::fs::read(&retained.path).unwrap(), b"echo retained");
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 1);
        drop(retained);
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
    }

    #[tokio::test]
    async fn cancelled_or_dropped_resolver_releases_its_own_directory() {
        for abort_future in [false, true] {
            let dir = TestDir::new();
            let server = TestServer::start(vec![Vec::new()], true);
            let urls = server.urls();
            let cache_root = dir.0.clone();
            let (tx, rx) = mpsc::channel(1);
            let task = tokio::spawn(async move {
                let client = reqwest::Client::builder().no_proxy()
                    .timeout(std::time::Duration::from_secs(30)).build().unwrap();
                download_resolved_script(ScriptKind::Sh,
                    &Pin { commit: None, branch: Some("main".into()) },
                    &cache_root, &client, &urls, &mut Some(rx)).await
            });
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while server.count() == 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
            }).await.expect("resolver did not reach the server");
            if abort_future {
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
            } else {
                tx.send(()).await.unwrap();
                let result = tokio::time::timeout(std::time::Duration::from_secs(3), task)
                    .await.expect("cancel did not release the resolver").unwrap();
                assert!(result.unwrap_err().to_string().contains("cancelled"));
            }
            assert_eq!(server.count(), 1, "cancel must not start the fallback");
            assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
        }
    }

    #[test]
    fn dropping_development_script_clones_preserves_the_checkout() {
        let dir = TestDir::new();
        let path = dir.0.join("install.ps1");
        std::fs::write(&path, b"Write-Host development").unwrap();
        let script = ResolvedScript {
            path: path.clone(),
            source: ScriptSource::DevCheckout,
            commit: None,
            branch: Some("main".into()),
            _cache_owner: None,
        };
        let clone = script.clone();
        drop(script);
        drop(clone);
        assert_eq!(std::fs::read(&path).unwrap(), b"Write-Host development");
    }

    #[tokio::test]
    async fn cancellation_interrupts_network_without_fallback_or_partial_publication() {
        // An already queued cancellation must not issue even the first GET.
        let dir = TestDir::new();
        let dest = dir.0.join("install.sh");
        std::fs::write(&dest, b"old installer").unwrap();
        let server = TestServer::new(vec![response(200, "text/plain", b"echo complete")]);
        let (tx, rx) = mpsc::channel(1);
        tx.send(()).await.unwrap();
        let result = download_from_urls_cancellable(ScriptKind::Sh, &dest,
            &test_client(), &server.urls(), &mut Some(rx)).await;
        assert!(result.unwrap_err().to_string().contains("cancelled"));
        assert_eq!(server.count(), 0);
        assert_eq!(std::fs::read(&dest).unwrap(), b"old installer");
        assert_only_destination(&dir, &dest);

        for headers_sent in [false, true] {
            let dir = TestDir::new();
            let dest = dir.0.join("install.sh");
            std::fs::write(&dest, b"old installer").unwrap();
            let partial = if headers_sent {
                b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\necho".to_vec()
            } else {
                Vec::new()
            };
            let server = TestServer::start(vec![partial], true);
            let urls = server.urls();
            let destination = dest.clone();
            let (tx, rx) = mpsc::channel(1);
            let task = tokio::spawn(async move {
                let client = reqwest::Client::builder().no_proxy()
                    .timeout(std::time::Duration::from_secs(30)).build().unwrap();
                download_from_urls_cancellable(ScriptKind::Sh, &destination,
                    &client, &urls, &mut Some(rx)).await
            });
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while server.count() == 0 {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                }
            }).await.expect("request did not reach the real server");
            tx.send(()).await.unwrap();
            // The server still owns the stalled socket until its Drop below;
            // cancellation must wake the client without peer cooperation.
            let result = tokio::time::timeout(std::time::Duration::from_secs(3), task)
                .await.expect("Cancel waited for the remote socket").unwrap();
            assert!(result.unwrap_err().to_string().contains("cancelled"));
            assert_eq!(server.count(), 1, "Cancel must not advance the fallback ladder");
            assert_eq!(std::fs::read(&dest).unwrap(), b"old installer");
            assert_only_destination(&dir, &dest);
        }
    }

    #[tokio::test]
    async fn truncated_successful_body_uses_next_source_and_preserves_ps1_bom() {
        let dir = TestDir::new();
        let dest = dir.0.join("install.ps1");
        std::fs::write(&dest, b"old installer").unwrap();
        let server = TestServer::new(vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\npartial".to_vec(),
            response(200, "text/plain", b"Write-Host 'complete'\n"),
        ]);
        download_from_urls(ScriptKind::Ps1, &dest, &test_client(), &server.urls()).await.unwrap();
        assert_eq!(server.count(), 2);
        assert_eq!(std::fs::read(&dest).unwrap(), prepare_cached_script_bytes(ScriptKind::Ps1, b"Write-Host 'complete'\n"));
        assert_only_destination(&dir, &dest);
    }

    #[tokio::test]
    async fn http_statuses_control_fallback_through_the_real_downloader() {
        for status in [403, 429, 500, 502, 503, 504, 401, 404, 410, 204, 206] {
            let dir = TestDir::new();
            let dest = dir.0.join("install.sh");
            std::fs::write(&dest, b"old installer").unwrap();
            let body = b"#!/bin/bash\necho complete\n";
            let server = TestServer::new(vec![response(status, "text/plain", b"unavailable"), response(200, "text/plain", body)]);
            let result = download_from_urls(ScriptKind::Sh, &dest, &test_client(), &server.urls()).await;
            if matches!(status, 403 | 429 | 500 | 502 | 503 | 504) {
                result.unwrap();
                assert_eq!(server.count(), 2, "HTTP {status}");
                assert_eq!(std::fs::read(&dest).unwrap(), body);
            } else {
                assert!(result.is_err(), "HTTP {status}");
                assert_eq!(server.count(), 1, "HTTP {status}");
                assert_eq!(std::fs::read(&dest).unwrap(), b"old installer");
            }
            assert_only_destination(&dir, &dest);
        }
    }

    #[tokio::test]
    async fn corrupt_and_error_bodies_never_replace_the_previous_installer() {
        for (content_type, body) in [
            ("text/plain", &b""[..]),
            ("text/plain", &b"\xef\xbb\xbf \r\n"[..]),
            ("text/plain", &b"\xff"[..]),
            ("text/plain", &b"script\0text"[..]),
            ("text/html", &b"unavailable"[..]),
            ("application/json", &b"{\"error\":true}"[..]),
            ("text/plain", &b" <!DOCTYPE html><html>unavailable</html>"[..]),
        ] {
            let dir = TestDir::new();
            let dest = dir.0.join("install.ps1");
            std::fs::write(&dest, b"old installer").unwrap();
            let server = TestServer::new(vec![response(200, content_type, body), response(503, "text/plain", b"unavailable")]);
            assert!(download_from_urls(ScriptKind::Ps1, &dest, &test_client(), &server.urls()).await.is_err());
            assert_eq!(server.count(), 2);
            assert_eq!(std::fs::read(&dest).unwrap(), b"old installer");
            assert_only_destination(&dir, &dest);
        }
    }

    #[tokio::test]
    async fn local_publish_failure_is_fatal_and_removes_the_temporary_script() {
        let dir = TestDir::new();
        let dest = dir.0.join("destination-is-a-directory");
        std::fs::create_dir(&dest).unwrap();
        let server = TestServer::new(vec![response(200, "text/plain", b"echo complete"), response(200, "text/plain", b"echo fallback")]);
        assert!(download_from_urls(ScriptKind::Sh, &dest, &test_client(), &server.urls()).await.is_err());
        assert_eq!(server.count(), 1, "local rename errors must not start another download");
        assert!(dest.is_dir());
        assert_only_destination(&dir, &dest);
    }

    #[tokio::test]
    async fn overlapping_downloads_publish_independently_without_shared_temporary_files() {
        let dir = TestDir::new();
        let dest = dir.0.join("install.sh");
        let body = b"#!/bin/bash\necho complete\n";
        let server = TestServer::new(vec![response(200, "text/plain", body), response(200, "text/plain", body)]);
        let client = test_client();
        let urls = server.urls();
        let (first, second) = tokio::join!(
            download_from_urls(ScriptKind::Sh, &dest, &client, &urls),
            download_from_urls(ScriptKind::Sh, &dest, &client, &urls),
        );
        first.unwrap();
        second.unwrap();
        assert_eq!(server.count(), 2);
        assert_eq!(std::fs::read(&dest).unwrap(), body);
        assert_only_destination(&dir, &dest);
    }

    #[tokio::test]
    async fn body_size_is_bounded_before_publication() {
        let dir = TestDir::new();
        let dest = dir.0.join("install.sh");
        std::fs::write(&dest, b"old installer").unwrap();
        let server = TestServer::new(vec![
            format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", MAX_SCRIPT_BYTES + 1).into_bytes(),
            response(200, "text/plain", b"echo complete"),
        ]);
        download_from_urls(ScriptKind::Sh, &dest, &test_client(), &server.urls()).await.unwrap();
        assert_eq!(server.count(), 2);
        assert_eq!(std::fs::read(&dest).unwrap(), b"echo complete");
        assert_only_destination(&dir, &dest);
    }

    #[tokio::test]
    async fn bodies_without_content_length_cannot_exceed_the_limit() {
        let dir = TestDir::new();
        let dest = dir.0.join("install.sh");
        let mut oversized = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n".to_vec();
        oversized.extend(std::iter::repeat(b'x').take(MAX_SCRIPT_BYTES + 1));
        let server = TestServer::new(vec![oversized, response(200, "text/plain", b"echo complete")]);
        download_from_urls(ScriptKind::Sh, &dest, &test_client(), &server.urls()).await.unwrap();
        assert_eq!(server.count(), 2);
        assert_eq!(std::fs::read(&dest).unwrap(), b"echo complete");
        assert_only_destination(&dir, &dest);
    }

    #[tokio::test]
    async fn production_client_refuses_cleartext_urls_without_mutating_the_cache() {
        let dir = TestDir::new();
        let dest = dir.0.join("install.sh");
        std::fs::write(&dest, b"old installer").unwrap();
        let server = TestServer::new(vec![response(200, "text/plain", b"echo complete")]);
        assert!(download_from_urls(ScriptKind::Sh, &dest, &download_client().unwrap(), &server.urls()).await.is_err());
        assert_eq!(server.count(), 0);
        assert_eq!(std::fs::read(&dest).unwrap(), b"old installer");
        assert_only_destination(&dir, &dest);
    }

    #[tokio::test]
    async fn redirect_loops_are_bounded_and_leave_the_destination_unchanged() {
        let dir = TestDir::new();
        let dest = dir.0.join("install.sh");
        std::fs::write(&dest, b"old installer").unwrap();
        let redirect = b"HTTP/1.1 302 Found\r\nLocation: /loop\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec();
        let server = TestServer::new(vec![redirect; 8]);
        let urls = vec![server.urls()[0].clone()];
        assert!(download_from_urls(ScriptKind::Sh, &dest, &test_client(), &urls).await.is_err());
        assert!(server.count() > 1 && server.count() <= 6, "redirect chain must stop at the configured bound");
        assert_eq!(std::fs::read(&dest).unwrap(), b"old installer");
        assert_only_destination(&dir, &dest);
    }

    #[test]
    fn valid_powershell_attribute_and_no_final_newline_are_accepted() {
        validate_script_body(b"[CmdletBinding()] param()", "text/plain").unwrap();
        validate_script_body(b"#!/bin/bash\necho complete", "text/plain").unwrap();
    }

    #[test]
    fn is_valid_commit_accepts_short_and_full_shas() {
        assert!(is_valid_commit("02d26981d3d4ad50e142399b8476f59ad5953ff0"));
        assert!(is_valid_commit("02d2698"));
        assert!(!is_valid_commit("02d269"));
        assert!(!is_valid_commit("not-a-sha"));
        assert!(!is_valid_commit(""));
    }

    #[test]
    fn prepare_cached_ps1_prefixes_utf8_bom() {
        let out = prepare_cached_script_bytes(ScriptKind::Ps1, b"Write-Host hi\n");
        assert!(out.starts_with(UTF8_BOM), "cached .ps1 must start with UTF-8 BOM");
        assert_eq!(&out[UTF8_BOM.len()..], b"Write-Host hi\n");
    }

    #[test]
    fn prepare_cached_ps1_does_not_double_bom() {
        let mut already = UTF8_BOM.to_vec();
        already.extend_from_slice(b"x");
        let out = prepare_cached_script_bytes(ScriptKind::Ps1, &already);
        assert_eq!(out, already);
        assert_eq!(out.windows(3).filter(|w| *w == UTF8_BOM).count(), 1);
    }

    #[test]
    fn prepare_cached_sh_stays_bomless() {
        let out = prepare_cached_script_bytes(ScriptKind::Sh, b"#!/usr/bin/env bash\n");
        assert!(!out.starts_with(UTF8_BOM));
        assert_eq!(out, b"#!/usr/bin/env bash\n");
    }

    #[test]
    fn commit_pins_are_distinguished_from_branch_pins() {
        assert!(is_valid_commit("02d26981d3d4ad50e142399b8476f59ad5953ff0"));
        assert!(!is_valid_commit("main"));
        assert!(!is_valid_commit("release/1.2.3"));
    }
}
