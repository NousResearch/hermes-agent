use std::{env, path::PathBuf, process::Command};

fn main() {
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
