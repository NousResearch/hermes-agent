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
    download(kind, &commit_or_ref, &dest).await?;
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

/// A cache filename must stay a single flat component inside the bootstrap
/// cache directory — the guard keeps every
/// cache filename a single flat component. `sanitize_ref` maps non-allowed chars to
/// `_`, so ref-embedded dots end up as literal filename characters
/// (`install-..ps1` is a harmless flat name) — the guard's job is to keep it
/// that way: no separators, no NUL, and no `..` path segment.
fn cache_filename_allowed(safe_name: &str) -> bool {
    !safe_name.is_empty()
        && !safe_name.contains(['/', '\\'])
        && !safe_name.contains('\0')
        && !safe_name
            .split(['/', '\\', '.'])
            .any(|seg| seg == "..")
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
async fn download(kind: ScriptKind, commit_or_ref: &str, dest_path: &Path) -> Result<()> {
    let url = format!(
        "https://raw.githubusercontent.com/NousResearch/hermes-agent/{}/scripts/{}",
        commit_or_ref,
        kind.filename()
    );

    // Security: the destination must stay inside the bootstrap
    // cache directory. `commit_or_ref` may be attacker-influenced, so a
    // crafted ref (`..`, `../x`) could otherwise make the download write
    // outside the cache. Defense in depth:
    //   1. lexical traversal rejection on the full path (before any mkdir),
    //   2. filename-level `..` rejection (post-sanitize loophole),
    //   3. canonicalize containment check on the resolved parent directory.
    if paths::contains_traversal(&dest_path.to_string_lossy()) {
        return Err(anyhow!(
            "download destination {} contains a path traversal sequence",
            dest_path.display()
        ));
    }
    let filename = dest_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| anyhow!("download destination {} has no file name", dest_path.display()))?;
    if !cache_filename_allowed(&filename) {
        return Err(anyhow!(
            "download destination {} contains a path traversal sequence",
            dest_path.display()
        ));
    }

    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("creating bootstrap-cache parent dir {}", parent.display())
        })?;
        let parent = parent
            .canonicalize()
            .with_context(|| format!("canonicalizing {}", parent.display()))?;
        let cache_dir = paths::bootstrap_cache_dir();
        std::fs::create_dir_all(&cache_dir).with_context(|| {
            format!("creating bootstrap-cache dir {}", cache_dir.display())
        })?;
        let canonical_cache = cache_dir
            .canonicalize()
            .with_context(|| format!("canonicalizing {}", cache_dir.display()))?;
        if !parent.starts_with(&canonical_cache) {
            return Err(anyhow!(
                "download destination {} resolves outside the bootstrap cache directory {}",
                dest_path.display(),
                canonical_cache.display()
            ));
        }
    }

    if let Some(parent) = dest_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("creating bootstrap-cache parent dir {}", parent.display())
        })?;
    }

    let tmp_path = dest_path.with_extension({
        let ext = dest_path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("tmp");
        format!("{ext}.tmp")
    });

    let response = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .context("building download client")?
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
    let bytes = prepare_cached_script_bytes(kind, &bytes);

    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .with_context(|| format!("creating temp file {}", tmp_path.display()))?;
    file.write_all(&bytes)
        .await
        .with_context(|| format!("writing temp file {}", tmp_path.display()))?;
    file.flush().await.context("flushing temp file")?;
    drop(file);

    tokio::fs::rename(&tmp_path, dest_path)
        .await
        .with_context(|| {
            format!(
                "renaming {} → {}",
                tmp_path.display(),
                dest_path.display()
            )
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

    // -- Path traversal hardening -----------------------------------------

    #[test]
    fn cache_filename_allowed_rejects_parent_dir_stems() {
        // `sanitize_ref` maps ref-embedded dots to literal filename chars, so
        // these stay flat, harmless names — but any name that WOULD carry a
        // `..` segment or separator must be rejected.
        assert!(!cache_filename_allowed("install-main/../evil.ps1"));
        assert!(!cache_filename_allowed("install-main\\..\\evil.ps1"));
        assert!(!cache_filename_allowed("a\0b.ps1"));
        // Ordinary refs remain allowed.
        assert!(cache_filename_allowed("install-main.ps1"));
        assert!(cache_filename_allowed("install-02d26981d3d4.ps1"));
        assert!(cache_filename_allowed("install-feature_x-1.2.sh"));
        // Refs that sanitize to dot-heavy flat names stay allowed.
        assert!(cache_filename_allowed("install-...ps1"));
        assert!(cache_filename_allowed("install-.._evil.ps1"));
    }

    #[test]
    fn download_rejects_traversal_destination() {
        // The lexical guard must fire before any network I/O or mkdir.
        let err = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                download(
                    ScriptKind::Ps1,
                    "main",
                    &PathBuf::from("/tmp/hermes-cache-escape/../../evil/install.ps1"),
                )
                .await
            })
            .expect_err("traversal destination must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("path traversal sequence"),
            "expected traversal rejection, got: {msg}"
        );
    }

    #[tokio::test]
    async fn download_rejects_destination_outside_cache_dir() {
        // A destination whose parent exists but is not inside
        // bootstrap_cache_dir() must be rejected by the canonical containment
        // check (this is the flagged write site).
        let outside = std::env::temp_dir().join(format!(
            "hermes-download-outside-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&outside).unwrap();
        let dest = outside.join("install-main.ps1");

        let err = download(ScriptKind::Ps1, "main", &dest)
            .await
            .expect_err("destination outside the cache dir must be rejected");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("outside the bootstrap cache directory"),
            "expected containment rejection, got: {msg}"
        );

        std::fs::remove_dir_all(&outside).unwrap();
    }
}
