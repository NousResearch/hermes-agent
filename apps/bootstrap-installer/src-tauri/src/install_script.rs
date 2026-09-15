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
    Cached,
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

/// Resolver cache plan for a pin that already has a local path computed.
///
/// Immutable commit pins reuse cache forever. Mutable branch/tag pins always
/// refresh, and only fall back to a stale cache when the refresh fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CachePlan {
    /// On-disk hit for an immutable pin — skip the network.
    Reuse,
    /// Download (or re-download). `stale_ok` means a failed refresh may return
    /// the existing cache file (mutable pins with a prior download).
    Fetch { stale_ok: bool },
}

pub(crate) fn cache_plan(immutable: bool, cached_exists: bool) -> CachePlan {
    if immutable && cached_exists {
        CachePlan::Reuse
    } else {
        CachePlan::Fetch {
            stale_ok: !immutable && cached_exists,
        }
    }
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
    // Commit SHAs are immutable — permanent cache reuse is safe.
    // Branch/tag pins are moving refs: always try to refresh so "Retry install"
    // cannot keep reusing a poisoned install-main.ps1 forever (#67193).
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

    let cached = cached_path(kind, &commit_or_ref);
    match cache_plan(immutable, cached.exists()) {
        CachePlan::Reuse => {
            emit_log(&format!(
                "[bootstrap] using cached {} for {}",
                kind.filename(),
                truncate_ref(&commit_or_ref)
            ));
            // Immutable pins are cached forever, so a .ps1 cached by a
            // pre-BOM-fix installer would keep the #67193 encoding bug on
            // every retry. Upgrade it in place before handing it out.
            upgrade_cached_script(kind, &cached, emit_log);
            return Ok(ResolvedScript {
                path: cached,
                source: ScriptSource::Cached,
                commit: pin.commit.clone(),
                branch: pin.branch.clone(),
            });
        }
        CachePlan::Fetch { stale_ok } => {
            emit_log(&format!(
                "[bootstrap] downloading {} for {} {} from GitHub",
                kind.filename(),
                if immutable {
                    "commit"
                } else {
                    "mutable ref"
                },
                truncate_ref(&commit_or_ref)
            ));

            match download(kind, &commit_or_ref, &cached).await {
                Ok(()) => {
                    emit_log(&format!("[bootstrap] cached to {}", cached.display()));
                    Ok(ResolvedScript {
                        path: cached,
                        source: ScriptSource::Downloaded,
                        commit: pin.commit.clone(),
                        branch: pin.branch.clone(),
                    })
                }
                Err(err) if stale_ok => {
                    emit_log(&format!(
                        "[bootstrap] WARNING: refresh failed for mutable ref {}; using stale cached {} at {}: {err:#}",
                        truncate_ref(&commit_or_ref),
                        kind.filename(),
                        cached.display()
                    ));
                    // Stale cache can predate the BOM fix too — upgrade it.
                    upgrade_cached_script(kind, &cached, emit_log);
                    Ok(ResolvedScript {
                        path: cached,
                        source: ScriptSource::Cached,
                        commit: pin.commit.clone(),
                        branch: pin.branch.clone(),
                    })
                }
                Err(err) => Err(err),
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Pin {
    pub commit: Option<String>,
    pub branch: Option<String>,
}

fn cached_path(kind: ScriptKind, commit_or_ref: &str) -> PathBuf {
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
/// are left unchanged — a BOM would break `#!/bin/bash`.
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

/// Upgrade a cached script written by a pre-BOM-fix installer in place.
///
/// `prepare_cached_script_bytes` only runs inside `download()`, but immutable
/// commit pins (and the stale-fallback path) reuse the on-disk file without
/// re-downloading — so a BOM-less `.ps1` cached before the #67193 fix would
/// keep reproducing the ANSI-codepage parse failure on every retry. Rewrites
/// through the same atomic tmp+rename shape as `download()`. Best-effort: a
/// failed upgrade logs a warning and keeps the original file (which is no
/// worse than the pre-existing behavior).
fn upgrade_cached_script(kind: ScriptKind, cached: &Path, emit_log: &impl Fn(&str)) {
    if !matches!(kind, ScriptKind::Ps1) {
        return;
    }
    let bytes = match std::fs::read(cached) {
        Ok(b) => b,
        Err(err) => {
            emit_log(&format!(
                "[bootstrap] WARNING: could not read cached script {} for BOM check: {err}",
                cached.display()
            ));
            return;
        }
    };
    if bytes.starts_with(UTF8_BOM) {
        return;
    }
    let upgraded = prepare_cached_script_bytes(kind, &bytes);
    let tmp = cached.with_extension("ps1.tmp");
    let result = std::fs::write(&tmp, &upgraded).and_then(|()| std::fs::rename(&tmp, cached));
    match result {
        Ok(()) => emit_log(&format!(
            "[bootstrap] upgraded cached {} with UTF-8 BOM (#67193)",
            cached.display()
        )),
        Err(err) => {
            let _ = std::fs::remove_file(&tmp);
            emit_log(&format!(
                "[bootstrap] WARNING: could not upgrade cached {} with UTF-8 BOM: {err}",
                cached.display()
            ));
        }
    }
}

/// Downloads to `dest_path` via reqwest with rustls. Atomically renames
/// `dest_path.tmp` → `dest_path` so partial writes don't poison the cache.
///
/// The client carries explicit timeouts: mutable branch pins call this on
/// EVERY run (#67193 cache-refresh fix), and the stale-cache fallback in
/// `resolve()` only fires when this returns `Err`. Without a timeout, a
/// black-holed connection (captive portal, hung proxy, silently dropped
/// packets) never errors — the whole bootstrap would hang here instead of
/// falling back to the cached script.
async fn download(kind: ScriptKind, commit_or_ref: &str, dest_path: &Path) -> Result<()> {
    let url = format!(
        "https://raw.githubusercontent.com/NousResearch/hermes-agent/{}/scripts/{}",
        commit_or_ref,
        kind.filename()
    );

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
        let out = prepare_cached_script_bytes(ScriptKind::Sh, b"#!/bin/bash\n");
        assert!(!out.starts_with(UTF8_BOM));
        assert_eq!(out, b"#!/bin/bash\n");
    }

    #[test]
    fn commit_pins_are_immutable_branch_pins_are_not() {
        // Mirrors the resolve() immutable decision: SHA pins may reuse cache
        // forever; branch pins must refresh so Retry cannot keep a bad script.
        assert!(is_valid_commit("02d26981d3d4ad50e142399b8476f59ad5953ff0"));
        assert!(!is_valid_commit("main"));
        assert!(!is_valid_commit("release/1.2.3"));
    }

    #[test]
    fn existing_branch_cache_plans_refresh_with_stale_fallback() {
        // Resolver-level: a prior install-main.ps1 must not short-circuit
        // Retry — mutable pins refresh, and only fall back if download fails.
        assert_eq!(
            cache_plan(/*immutable=*/ false, /*cached_exists=*/ true),
            CachePlan::Fetch { stale_ok: true }
        );
        assert_eq!(
            cache_plan(/*immutable=*/ true, /*cached_exists=*/ true),
            CachePlan::Reuse
        );
        assert_eq!(
            cache_plan(/*immutable=*/ false, /*cached_exists=*/ false),
            CachePlan::Fetch { stale_ok: false }
        );
        assert_eq!(
            cache_plan(/*immutable=*/ true, /*cached_exists=*/ false),
            CachePlan::Fetch { stale_ok: false }
        );
    }

    #[test]
    fn upgrade_cached_script_adds_bom_to_legacy_ps1() {
        // A .ps1 cached by a pre-#67193 installer has no BOM; the Reuse path
        // must upgrade it in place instead of serving the broken bytes forever.
        let dir = std::env::temp_dir().join(format!("hermes-bom-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cached = dir.join("install-abc1234.ps1");
        std::fs::write(&cached, b"Write-Host legacy\n").unwrap();

        upgrade_cached_script(ScriptKind::Ps1, &cached, &|_| {});
        let bytes = std::fs::read(&cached).unwrap();
        assert!(bytes.starts_with(UTF8_BOM), "legacy cache must gain a BOM");
        assert_eq!(&bytes[UTF8_BOM.len()..], b"Write-Host legacy\n");

        // Idempotent: a second pass must not double the BOM.
        upgrade_cached_script(ScriptKind::Ps1, &cached, &|_| {});
        let again = std::fs::read(&cached).unwrap();
        assert_eq!(again, bytes);

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn upgrade_cached_script_leaves_sh_untouched() {
        let dir = std::env::temp_dir().join(format!("hermes-bom-sh-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cached = dir.join("install-main.sh");
        std::fs::write(&cached, b"#!/bin/bash\n").unwrap();

        upgrade_cached_script(ScriptKind::Sh, &cached, &|_| {});
        assert_eq!(std::fs::read(&cached).unwrap(), b"#!/bin/bash\n");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
