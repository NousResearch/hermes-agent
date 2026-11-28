use std::{env, path::PathBuf, process::Command};

fn main() {
    // Windows: embed a Common Controls v6 manifest dependency in every
    // linked binary, unit-test harness included. Without it, imports that resolve into
    // comctl32 v6 fail at load with STATUS_DLL_INIT_FAILED (0xc0000142).
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") && env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        println!("cargo:rustc-link-arg=/MANIFEST:EMBED");
        println!("cargo:rustc-link-arg=/MANIFESTDEPENDENCY:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'");
    }
    println!("cargo:rerun-if-env-changed=HERMES_DESKTOP_VERSION");
    println!("cargo:rerun-if-env-changed=HERMES_DESKTOP_SOURCE_REVISION");
    for key in ["HERMES_DESKTOP_VERSION", "HERMES_DESKTOP_SOURCE_REVISION"] {
        if let Ok(value) = env::var(key) {
            if !value.contains(['\n', '\r']) {
                println!("cargo:rustc-env={key}={value}");
            }
        }
    }
    if env::var_os("HERMES_DESKTOP_SOURCE_REVISION").is_some() {
        return;
    }
    let Some(root) = env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .and_then(|path| path.ancestors().nth(4).map(|path| path.to_owned()))
        .and_then(|path| path.canonicalize().ok())
    else {
        return;
    };
    let git = |args: &[&str]| {
        let output = Command::new("git").arg("-C").arg(&root).args(args).output().ok()?;
        output.status.success().then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    if git(&["rev-parse", "--show-toplevel"])
        .and_then(|path| PathBuf::from(path).canonicalize().ok())
        .as_ref() != Some(&root)
    {
        return;
    }
    println!("cargo:rerun-if-changed={}", root.join(".git/HEAD").display());
    if let Some(revision) = git(&["rev-parse", "HEAD"])
        .filter(|value| value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
    {
        println!("cargo:rustc-env=HERMES_DESKTOP_SOURCE_REVISION={revision}");
    }
}
