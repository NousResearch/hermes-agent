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
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::paths;

const RAW_GITHUB_BASE: &str = "https://raw.githubusercontent.com/NousResearch/hermes-agent";
const DOWNLOAD_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const DOWNLOAD_READ_TIMEOUT: Duration = Duration::from_secs(20);
const DOWNLOAD_TOTAL_TIMEOUT: Duration = Duration::from_secs(30);

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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptSource {
    DevCheckout,
    Bundled,
    Cached,
    CachedFallback,
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
) -> Result<ResolvedScript> {
    // 1. Dev shortcut.
    if let Ok(repo_root) = std::env::var("HERMES_SETUP_DEV_REPO_ROOT") {
        let candidate = PathBuf::from(repo_root)
            .join("scripts")
            .join(kind.filename());
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
            });
        }
    }

    // 2. (Not implemented) bundled fallback.

    resolve_network(
        kind,
        pin,
        &paths::bootstrap_cache_dir(),
        RAW_GITHUB_BASE,
        emit_log,
    )
    .await
}

async fn resolve_network(
    kind: ScriptKind,
    pin: &Pin,
    cache_dir: &Path,
    raw_base_url: &str,
    emit_log: &impl Fn(&str),
) -> Result<ResolvedScript> {
    let client = download_client(
        DOWNLOAD_CONNECT_TIMEOUT,
        DOWNLOAD_READ_TIMEOUT,
        DOWNLOAD_TOTAL_TIMEOUT,
    )?;
    resolve_network_with_client(kind, pin, cache_dir, raw_base_url, emit_log, &client).await
}

async fn resolve_network_with_client(
    kind: ScriptKind,
    pin: &Pin,
    cache_dir: &Path,
    raw_base_url: &str,
    emit_log: &impl Fn(&str),
    client: &reqwest::Client,
) -> Result<ResolvedScript> {
    // Only a valid commit is immutable. Branches and tags are moving refs and
    // therefore must be refreshed before each install/update run.
    let (commit_or_ref, immutable) = match (&pin.commit, &pin.branch) {
        (Some(c), _) if is_valid_commit(c) => (c.clone(), true),
        (_, Some(b)) if !b.trim().is_empty() => (b.clone(), false),
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

    let cached = cached_path_in(cache_dir, kind, &commit_or_ref);
    if immutable && cached.exists() {
        emit_log(&format!(
            "[bootstrap] using immutable cached {} for commit {}",
            kind.filename(),
            truncate_ref(&commit_or_ref)
        ));
        return Ok(ResolvedScript {
            path: cached,
            source: ScriptSource::Cached,
            commit: pin.commit.clone(),
            branch: pin.branch.clone(),
        });
    }

    emit_log(&format!(
        "[bootstrap] downloading {} for {} {} from GitHub",
        kind.filename(),
        if immutable { "commit" } else { "mutable ref" },
        truncate_ref(&commit_or_ref)
    ));

    match download(
        kind,
        &commit_or_ref,
        &cached,
        raw_base_url.trim_end_matches('/'),
        client,
    )
    .await
    {
        Ok(()) => {
            emit_log(&format!("[bootstrap] cached to {}", cached.display()));
            Ok(ResolvedScript {
                path: cached,
                source: ScriptSource::Downloaded,
                commit: pin.commit.clone(),
                branch: pin.branch.clone(),
            })
        }
        Err(err) if !immutable && cached.exists() => {
            emit_log(&format!(
                "[bootstrap] WARNING: refresh failed for mutable ref {}; using stale cached {} at {} as explicit network fallback: {err:#}",
                truncate_ref(&commit_or_ref),
                kind.filename(),
                cached.display()
            ));
            Ok(ResolvedScript {
                path: cached,
                source: ScriptSource::CachedFallback,
                commit: pin.commit.clone(),
                branch: pin.branch.clone(),
            })
        }
        Err(err) => Err(err).with_context(|| {
            if immutable {
                format!(
                    "no cached install script is available for immutable commit {}",
                    truncate_ref(&commit_or_ref)
                )
            } else {
                format!(
                    "refresh failed for mutable ref {} and no cached fallback is available",
                    truncate_ref(&commit_or_ref)
                )
            }
        }),
    }
}

#[derive(Debug, Clone, Default)]
pub struct Pin {
    pub commit: Option<String>,
    pub branch: Option<String>,
}

fn cached_path_in(cache_dir: &Path, kind: ScriptKind, commit_or_ref: &str) -> PathBuf {
    let safe = sanitize_ref(commit_or_ref);
    let filename = match kind {
        ScriptKind::Ps1 => format!("install-{safe}.ps1"),
        ScriptKind::Sh => format!("install-{safe}.sh"),
    };
    cache_dir.join(filename)
}

/// Replace anything that's not [A-Za-z0-9._-] with `_`. Branch refs can
/// contain `/`, dots, etc.; we want a flat filename.
fn sanitize_ref(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn truncate_ref(s: &str) -> &str {
    if is_valid_commit(s) && s.len() >= 12 {
        &s[..12]
    } else {
        s
    }
}

/// Downloads to `dest_path` via reqwest with rustls. Atomically renames
/// `dest_path.tmp` → `dest_path` so partial writes don't poison the cache.
async fn download(
    kind: ScriptKind,
    commit_or_ref: &str,
    dest_path: &Path,
    raw_base_url: &str,
    client: &reqwest::Client,
) -> Result<()> {
    let url = format!(
        "{}/{}/scripts/{}",
        raw_base_url,
        commit_or_ref,
        kind.filename()
    );

    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating bootstrap-cache parent dir {}", parent.display()))?;
    }

    let response = client
        .get(&url)
        .header("User-Agent", "hermes-setup/0.0.1")
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;

    if !response.status().is_success() {
        return Err(anyhow!(
            "Failed to download {}: HTTP {} from {}",
            kind.filename(),
            response.status(),
            url
        ));
    }

    let bytes = response
        .bytes()
        .await
        .with_context(|| format!("reading body of {url}"))?;

    let tmp_path = dest_path.with_extension({
        let ext = dest_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("tmp");
        format!("{ext}.{}.tmp", Uuid::new_v4())
    });

    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("creating temp file {}", tmp_path.display()))?;
    if let Err(err) = file.write_all(&bytes).await {
        drop(file);
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(err).with_context(|| format!("writing temp file {}", tmp_path.display()));
    }
    if let Err(err) = file.flush().await {
        drop(file);
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(err).context("flushing temp file");
    }
    drop(file);

    if let Err(err) = atomic_replace(&tmp_path, dest_path).await {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(err);
    }

    Ok(())
}

fn download_client(
    connect_timeout: Duration,
    read_timeout: Duration,
    total_timeout: Duration,
) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        .read_timeout(read_timeout)
        .timeout(total_timeout)
        .build()
        .context("building install-script HTTP client")
}

#[cfg(not(target_os = "windows"))]
async fn atomic_replace(tmp_path: &Path, dest_path: &Path) -> Result<()> {
    tokio::fs::rename(tmp_path, dest_path)
        .await
        .with_context(|| {
            format!(
                "atomically renaming {} → {}",
                tmp_path.display(),
                dest_path.display()
            )
        })
}

#[cfg(target_os = "windows")]
async fn atomic_replace(tmp_path: &Path, dest_path: &Path) -> Result<()> {
    use std::os::windows::ffi::OsStrExt;

    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;

    #[link(name = "kernel32")]
    extern "system" {
        fn MoveFileExW(
            existing_file_name: *const u16,
            new_file_name: *const u16,
            flags: u32,
        ) -> i32;
    }

    let src: Vec<u16> = tmp_path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let dst: Vec<u16> = dest_path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    let moved = unsafe {
        MoveFileExW(
            src.as_ptr(),
            dst.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "atomically replacing {} with {}",
                dest_path.display(),
                tmp_path.display()
            )
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    #[test]
    fn is_valid_commit_accepts_short_and_full_shas() {
        assert!(is_valid_commit("02d26981d3d4ad50e142399b8476f59ad5953ff0"));
        assert!(is_valid_commit("02d2698"));
        assert!(!is_valid_commit("02d269"));
        assert!(!is_valid_commit("not-a-sha"));
        assert!(!is_valid_commit(""));
    }

    #[test]
    fn sanitize_ref_replaces_slashes() {
        assert_eq!(sanitize_ref("bb/gui"), "bb_gui");
        assert_eq!(sanitize_ref("main"), "main");
        assert_eq!(sanitize_ref("release/1.2.3"), "release_1.2.3");
    }

    fn test_cache_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hermes-install-script-{}", Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn serve_once(status: &str, body: &[u8], declared_length: Option<usize>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let status = status.to_string();
        let body = body.to_vec();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request);
            let content_length = declared_length.unwrap_or(body.len());
            write!(
                stream,
                "HTTP/1.1 {status}\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
            )
            .unwrap();
            stream.write_all(&body).unwrap();
        });
        format!("http://{addr}")
    }

    fn serve_silent_once() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (_stream, _) = listener.accept().unwrap();
            thread::sleep(Duration::from_secs(5));
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn immutable_commit_reuses_cache_without_network() {
        let cache_dir = test_cache_dir();
        let pin = Pin {
            commit: Some("02d26981d3d4".into()),
            branch: Some("main".into()),
        };
        let cached = cached_path_in(&cache_dir, ScriptKind::Ps1, pin.commit.as_ref().unwrap());
        std::fs::write(&cached, b"cached commit").unwrap();
        let logs = std::sync::Mutex::new(Vec::new());

        let resolved = resolve_network(
            ScriptKind::Ps1,
            &pin,
            &cache_dir,
            "http://127.0.0.1:9",
            &|line| logs.lock().unwrap().push(line.to_string()),
        )
        .await
        .unwrap();

        assert_eq!(resolved.source, ScriptSource::Cached);
        assert_eq!(std::fs::read(&cached).unwrap(), b"cached commit");
        assert!(logs.lock().unwrap().join("\n").contains("immutable cached"));
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn mutable_ref_refreshes_and_atomically_replaces_old_cache() {
        let cache_dir = test_cache_dir();
        let pin = Pin {
            commit: None,
            branch: Some("main".into()),
        };
        let cached = cached_path_in(&cache_dir, ScriptKind::Ps1, "main");
        std::fs::write(&cached, b"old script").unwrap();
        let server = serve_once("200 OK", b"new script", None);

        let resolved = resolve_network(ScriptKind::Ps1, &pin, &cache_dir, &server, &|_| {})
            .await
            .unwrap();

        assert_eq!(resolved.source, ScriptSource::Downloaded);
        assert_eq!(std::fs::read(&cached).unwrap(), b"new script");
        assert_eq!(
            std::fs::read_dir(&cache_dir).unwrap().count(),
            1,
            "temporary download must not remain in the cache"
        );
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn mutable_ref_uses_explicit_stale_fallback_on_download_failure() {
        let cache_dir = test_cache_dir();
        let pin = Pin {
            commit: None,
            branch: Some("main".into()),
        };
        let cached = cached_path_in(&cache_dir, ScriptKind::Ps1, "main");
        std::fs::write(&cached, b"old script").unwrap();
        let server = serve_once("503 Service Unavailable", b"", None);
        let logs = std::sync::Mutex::new(Vec::new());

        let resolved = resolve_network(ScriptKind::Ps1, &pin, &cache_dir, &server, &|line| {
            logs.lock().unwrap().push(line.to_string())
        })
        .await
        .unwrap();

        assert_eq!(resolved.source, ScriptSource::CachedFallback);
        assert_eq!(std::fs::read(&cached).unwrap(), b"old script");
        let joined = logs.lock().unwrap().join("\n");
        assert!(joined.contains("WARNING"));
        assert!(joined.contains("explicit network fallback"));
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn mutable_ref_without_cache_fails_closed() {
        let cache_dir = test_cache_dir();
        let pin = Pin {
            commit: None,
            branch: Some("main".into()),
        };
        let server = serve_once("503 Service Unavailable", b"", None);

        let err = resolve_network(ScriptKind::Ps1, &pin, &cache_dir, &server, &|_| {})
            .await
            .unwrap_err();

        assert!(format!("{err:#}").contains("no cached fallback is available"));
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 0);
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn interrupted_mutable_download_does_not_poison_old_cache() {
        let cache_dir = test_cache_dir();
        let pin = Pin {
            commit: None,
            branch: Some("main".into()),
        };
        let cached = cached_path_in(&cache_dir, ScriptKind::Ps1, "main");
        std::fs::write(&cached, b"known-good script").unwrap();
        let server = serve_once("200 OK", b"partial", Some(128));

        let resolved = resolve_network(ScriptKind::Ps1, &pin, &cache_dir, &server, &|_| {})
            .await
            .unwrap();

        assert_eq!(resolved.source, ScriptSource::CachedFallback);
        assert_eq!(std::fs::read(&cached).unwrap(), b"known-good script");
        assert_eq!(std::fs::read_dir(&cache_dir).unwrap().count(), 1);
        let _ = std::fs::remove_dir_all(cache_dir);
    }

    #[tokio::test]
    async fn silent_server_times_out_and_preserves_mutable_fallback() {
        let cache_dir = test_cache_dir();
        let pin = Pin {
            commit: None,
            branch: Some("main".into()),
        };
        let cached = cached_path_in(&cache_dir, ScriptKind::Ps1, "main");
        std::fs::write(&cached, b"known-good script").unwrap();
        let server = serve_silent_once();
        let client = download_client(
            Duration::from_millis(100),
            Duration::from_millis(200),
            Duration::from_millis(300),
        )
        .unwrap();
        let resolved = tokio::time::timeout(
            Duration::from_secs(2),
            resolve_network_with_client(
                ScriptKind::Ps1,
                &pin,
                &cache_dir,
                &server,
                &|_| {},
                &client,
            ),
        )
        .await
        .expect("silent endpoint must not block the installer")
        .unwrap();

        assert_eq!(resolved.source, ScriptSource::CachedFallback);
        assert_eq!(std::fs::read(&cached).unwrap(), b"known-good script");
        let _ = std::fs::remove_dir_all(cache_dir);
    }
}
