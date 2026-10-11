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
use tokio::io::AsyncWriteExt;

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

/// One try plus two retries. First-run installs have no stale cache, so a
/// single GitHub raw 503 used to abort the whole beta-to-stable setup (#88475).
const MAX_DOWNLOAD_ATTEMPTS: u32 = 3;

/// GitHub raw / CDN blips we should swallow. 404/401/403 stay fatal.
pub(crate) fn is_transient_script_http_status(code: u16) -> bool {
    matches!(code, 408 | 429 | 500 | 502 | 503 | 504)
}

pub(crate) fn should_retry_script_download(
    http_status: Option<u16>,
    transport_error: bool,
    attempt: u32,
) -> bool {
    should_retry_script_download_for_ref(http_status, transport_error, attempt, /*immutable=*/ true)
}

/// Retry decision for one download attempt of the install script.
///
/// `immutable` is true for a commit-pinned fetch, false for a mutable branch
/// ref (`main`). On the mutable path a 404 is retried once: a raw-CDN cache
/// purge right after a push can briefly 404 a file that exists (#121615), and
/// a first-run install has no stale cache to fall back to. A bad immutable
/// pin must still look fatal immediately — a 404 there never burns retries.
pub(crate) fn should_retry_script_download_for_ref(
    http_status: Option<u16>,
    transport_error: bool,
    attempt: u32,
    immutable: bool,
) -> bool {
    if attempt >= MAX_DOWNLOAD_ATTEMPTS {
        return false;
    }
    if transport_error {
        return true;
    }
    match http_status {
        Some(404) if !immutable => true,
        Some(status) => is_transient_script_http_status(status),
        None => false,
    }
}

fn download_backoff(attempt: u32) -> std::time::Duration {
    let shift = attempt.saturating_sub(1).min(3);
    std::time::Duration::from_millis(200 * (1u64 << shift))
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

    let dest = download_path(kind, &commit_or_ref);
    emit_log(&format!(
        "[bootstrap] downloading {} for {} from GitHub",
        kind.filename(),
        truncate_ref(&commit_or_ref)
    ));
    // A 404 is retried on the mutable-ref path only (#121615): a raw-CDN cache
    // purge right after a push can briefly 404 an existing file. Commit pins
    // stay fatal on 404 — a bad pin must not look like a blip.
    let immutable = is_valid_commit(&commit_or_ref);
    download(kind, &commit_or_ref, &dest, immutable).await?;
    emit_log(&format!("[bootstrap] downloaded to {}", dest.display()));
    Ok(ResolvedScript {
        path: dest,
        source: ScriptSource::Downloaded,
        commit: pin.commit.clone(),
        branch: pin.branch.clone(),
    })
}

#[derive(Debug, Clone, Default)]
pub struct Pin {
    pub commit: Option<String>,
    pub branch: Option<String>,
}

fn download_path(kind: ScriptKind, commit_or_ref: &str) -> PathBuf {
    let safe = sanitize_ref(commit_or_ref);
    let filename = match kind {
        ScriptKind::Ps1 => format!("install-{safe}.ps1"),
        ScriptKind::Sh => format!("install-{safe}.sh"),
    };
    paths::bootstrap_cache_dir().join(filename)
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

/// Downloads to `dest_path` via reqwest with rustls. Atomically renames
/// `dest_path.tmp` → `dest_path` so a partial write is never executed.
///
/// Explicit timeouts: this runs on every bootstrap, and a black-holed
/// connection (captive portal, hung proxy) would otherwise hang forever
/// instead of failing so the user can Retry.
async fn download(
    kind: ScriptKind,
    commit_or_ref: &str,
    dest_path: &Path,
    immutable: bool,
) -> Result<()> {
    let url = format!(
        "https://raw.githubusercontent.com/NousResearch/hermes-agent/{}/scripts/{}",
        commit_or_ref,
        kind.filename()
    );

    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating bootstrap-cache parent dir {}", parent.display()))?;
    }

    let tmp_path = dest_path.with_extension({
        let ext = dest_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("tmp");
        format!("{ext}.tmp")
    });

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .context("building download client")?;

    let mut last_err: Option<anyhow::Error> = None;
    for attempt in 1..=MAX_DOWNLOAD_ATTEMPTS {
        match download_once(&client, kind, &url, &tmp_path, dest_path).await {
            Ok(()) => return Ok(()),
            Err(err) => {
                let retry = should_retry_script_download_for_ref(
                    err.http_status,
                    err.transport,
                    attempt,
                    immutable,
                );
                last_err = Some(err.err);
                let _ = tokio::fs::remove_file(&tmp_path).await;
                if !retry {
                    break;
                }
                tokio::time::sleep(download_backoff(attempt)).await;
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow!("download failed")))
}

struct DownloadAttemptErr {
    err: anyhow::Error,
    http_status: Option<u16>,
    transport: bool,
}

async fn download_once(
    client: &reqwest::Client,
    kind: ScriptKind,
    url: &str,
    tmp_path: &Path,
    dest_path: &Path,
) -> Result<(), DownloadAttemptErr> {
    let response = match client
        .get(url)
        .header("User-Agent", "hermes-setup/0.0.1")
        .send()
        .await
    {
        Ok(response) => response,
        Err(err) => {
            let transport = err.is_timeout() || err.is_connect() || err.is_request();
            return Err(DownloadAttemptErr {
                err: anyhow!(err).context(format!("GET {url}")),
                http_status: None,
                transport,
            });
        }
    };

    if !response.status().is_success() {
        let http_status = response.status().as_u16();
        return Err(DownloadAttemptErr {
            err: anyhow!(
                "Failed to download {}: HTTP {} from {}",
                kind.filename(),
                response.status(),
                url
            ),
            http_status: Some(http_status),
            transport: false,
        });
    }

    let bytes = response.bytes().await.map_err(|err| DownloadAttemptErr {
        transport: err.is_timeout() || err.is_connect() || err.is_request(),
        err: anyhow!(err).context(format!("reading body of {url}")),
        http_status: None,
    })?;
    let bytes = prepare_cached_script_bytes(kind, &bytes);

    let mut file = tokio::fs::File::create(tmp_path)
        .await
        .map_err(|err| DownloadAttemptErr {
            err: anyhow!(err).context(format!("creating temp file {}", tmp_path.display())),
            http_status: None,
            transport: false,
        })?;
    file.write_all(&bytes)
        .await
        .map_err(|err| DownloadAttemptErr {
            err: anyhow!(err).context(format!("writing temp file {}", tmp_path.display())),
            http_status: None,
            transport: false,
        })?;
    file.flush().await.map_err(|err| DownloadAttemptErr {
        err: anyhow!(err).context("flushing temp file"),
        http_status: None,
        transport: false,
    })?;
    drop(file);

    tokio::fs::rename(tmp_path, dest_path)
        .await
        .map_err(|err| DownloadAttemptErr {
            err: anyhow!(err).context(format!(
                "renaming {} → {}",
                tmp_path.display(),
                dest_path.display()
            )),
            http_status: None,
            transport: false,
        })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn prepare_cached_ps1_prefixes_utf8_bom() {
        let out = prepare_cached_script_bytes(ScriptKind::Ps1, b"Write-Host hi\n");
        assert!(
            out.starts_with(UTF8_BOM),
            "cached .ps1 must start with UTF-8 BOM"
        );
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

    #[test]
    fn transient_script_http_statuses_are_the_cdn_blips() {
        for code in [408, 429, 500, 502, 503, 504] {
            assert!(is_transient_script_http_status(code), "{code} should retry");
        }
        for code in [200, 301, 400, 401, 403, 404, 410] {
            assert!(
                !is_transient_script_http_status(code),
                "{code} must stay fatal"
            );
        }
    }

    #[test]
    fn script_download_retries_are_bounded_and_skip_404() {
        assert!(should_retry_script_download(Some(503), false, 1));
        assert!(should_retry_script_download(Some(502), false, 2));
        assert!(
            !should_retry_script_download(Some(503), false, MAX_DOWNLOAD_ATTEMPTS),
            "third failure is terminal"
        );
        assert!(
            !should_retry_script_download(Some(404), false, 1),
            "a bad immutable pin must not burn retries"
        );
        assert!(!should_retry_script_download(Some(401), false, 1));
        assert!(should_retry_script_download(None, true, 1));
        assert!(!should_retry_script_download(
            None,
            true,
            MAX_DOWNLOAD_ATTEMPTS
        ));
        assert!(!should_retry_script_download(None, false, 1));
    }

    #[test]
    fn mutable_ref_404_is_retried_but_stays_bounded() {
        // #121615: raw-CDN cache purges can briefly 404 an existing file on a
        // moving branch. The mutable-ref path gets one bounded retry; the
        // third attempt stays terminal either way.
        assert!(should_retry_script_download_for_ref(
            Some(404),
            false,
            1,
            /*immutable=*/ false
        ));
        assert!(!should_retry_script_download_for_ref(
            Some(404),
            false,
            1,
            /*immutable=*/ true
        ));
        assert!(!should_retry_script_download_for_ref(
            Some(404),
            false,
            MAX_DOWNLOAD_ATTEMPTS,
            /*immutable=*/ false
        ));
    }
}
