//! Catalog plugins the Rust runtime can run: Agent Plugins v1 packages
//! (plugin.json + mcp.json + skills/), the format hermes_cli/agent_plugins.py
//! loads. `install` clones the catalog's repository at its exact pinned SHA
//! (README rule 2) and copies the entry's subdir to <root>/plugins/<name>;
//! its skills join the skill store and its MCP servers (streamable-http or
//! stdio) are connected for every turn while the plugin is enabled. Python
//! (`plugin.yaml` + `__init__.py`) plugins run in the installed Hermes Python
//! agent: assets/python_plugin_bridge.py loads the plugin with Hermes's own
//! PluginManager and serves the tools it registered over MCP stdio.
use crate::mcp::{Mcp, Stdio};
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, path::{Component, Path, PathBuf}, process::Command};

const PLUGIN_SCHEMA: &str = "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json";
const MCP_SCHEMA: &str = "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json";

/// One catalog entry, as the plugin-catalog YAML pins it.
#[derive(Clone, Debug, Default)]
pub struct CatalogEntry { pub name: String, pub title: String, pub repo: String, pub sha: String, pub subdir: String }

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Installed {
    pub name: String, pub title: String, pub version: String, pub description: String, pub sha: String,
    pub enabled: bool, pub servers: Vec<String>, pub skills: Vec<String>,
}

fn dir(root: &Path) -> PathBuf { root.join("plugins") }
fn index(root: &Path) -> PathBuf { dir(root).join("installed.json") }
pub fn installed(root: &Path) -> Result<Vec<Installed>> {
    match fs::read(index(root)) {
        Ok(bytes) => Ok(serde_json::from_slice(&bytes).context("Invalid plugins/installed.json")?),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
        Err(e) => Err(e.into()),
    }
}
fn save(root: &Path, list: &[Installed]) -> Result<()> {
    fs::create_dir_all(dir(root))?;
    let temp = tempfile::NamedTempFile::new_in(dir(root))?;
    serde_json::to_writer_pretty(temp.as_file(), list)?;
    temp.persist(index(root))?;
    Ok(())
}

fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 64 && !name.contains("--") && !name.contains("..")
        && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'.')
        && name.bytes().next().is_some_and(|b| b.is_ascii_alphanumeric())
}
fn inside(base: &Path, relative: &str) -> Option<PathBuf> {
    let path = Path::new(relative);
    (!relative.is_empty() && path.components().all(|c| matches!(c, Component::Normal(_)))).then(|| base.join(path))
}

/// Clone `entry` at its pinned commit and install it. Blocking (git): run off the UI thread.
pub fn install(root: &Path, entry: &CatalogEntry) -> Result<Installed> {
    ensure!(valid_name(&entry.name), "That plugin name is not valid");
    ensure!(entry.sha.len() == 40 && entry.sha.bytes().all(|b| b.is_ascii_hexdigit()), "The catalog entry has no exact commit pin");
    let repo = reqwest::Url::parse(&entry.repo).context("The catalog entry has no valid repository")?;
    ensure!(repo.scheme() == "https" && repo.username().is_empty() && repo.password().is_none(), "Plugins install only from an https repository");
    fs::create_dir_all(dir(root))?;
    let staging = tempfile::tempdir_in(dir(root))?;
    let git = |args: &[&str]| -> Result<()> {
        let out = Command::new("git").args(args).current_dir(staging.path()).env("GIT_TERMINAL_PROMPT", "0").output().context("git is not available on this computer")?;
        ensure!(out.status.success(), "git {} failed: {}", args.first().unwrap_or(&""), String::from_utf8_lossy(&out.stderr).trim());
        Ok(())
    };
    git(&["init", "-q"])?;
    git(&["fetch", "-q", "--depth", "1", repo.as_str(), &entry.sha])?;
    git(&["checkout", "-q", "FETCH_HEAD"])?;
    let source = if entry.subdir.trim().is_empty() { staging.path().to_path_buf() } else { inside(staging.path(), entry.subdir.trim()).context("The catalog subdir is not valid")? };
    let package = read_package(&source)?;
    if package.servers.contains_key("python") && python_hermes().is_none() {
        bail!("This is a Python Hermes plugin. It runs through the Hermes Python agent, which is not installed on this computer (install it from hermes-agent.nousresearch.com, then try again).");
    }
    let target = dir(root).join(&entry.name);
    if target.exists() { fs::remove_dir_all(&target)?; }
    copy_tree(&source, &target)?;
    fs::create_dir_all(root.join("plugin-data").join(&entry.name))?;
    let docs = package.skills.iter().map(|(skill, doc)| (format!("plugin/{}/{skill}", entry.name), doc.clone())).collect();
    crate::skill_catalog::put_documents(root, &format!("plugin/{}/", entry.name), docs)?;
    let record = Installed { name: entry.name.clone(), title: if entry.title.is_empty() { package.name.clone() } else { entry.title.clone() },
        version: package.version, description: package.description, sha: entry.sha.clone(), enabled: true,
        servers: package.servers.keys().cloned().collect(), skills: package.skills.into_iter().map(|(s, _)| s).collect() };
    let mut list = installed(root)?;
    list.retain(|p| p.name != entry.name);
    list.push(record.clone());
    save(root, &list)?;
    Ok(record)
}

pub fn set_enabled(root: &Path, name: &str, on: bool) -> Result<()> {
    let mut list = installed(root)?;
    let plugin = list.iter_mut().find(|p| p.name == name).context("That plugin is not installed")?;
    plugin.enabled = on;
    let ids: Vec<String> = plugin.skills.iter().map(|s| format!("plugin/{name}/{s}")).collect();
    save(root, &list)?;
    for id in ids { crate::skill_catalog::enable(root, &id, on)?; }
    Ok(())
}

/// The streamable-http servers of an installed plugin, for sign-in: (server name, url).
pub fn http_servers(root: &Path, name: &str) -> Result<Vec<(String, String)>> {
    let package = read_package(&dir(root).join(name))?;
    Ok(package.servers.into_iter().filter_map(|(n, s)| match s { Server::Http { url, .. } => Some((n, url)), _ => None }).collect())
}
/// Browser sign-in for every HTTP server of a plugin (MCP OAuth 2.1).
pub async fn sign_in(root: &Path, name: &str, open: impl Fn(&str) + Clone) -> Result<usize> {
    ensure!(valid_name(name), "That plugin name is not valid");
    let data = root.join("plugin-data").join(name);
    let servers = http_servers(root, name)?;
    ensure!(!servers.is_empty(), "This plugin has no servers that sign in");
    for (server, url) in &servers { crate::mcp_oauth::sign_in(&data, server, url, open.clone()).await.with_context(|| format!("{server}"))?; }
    Ok(servers.len())
}
pub fn signed_in(root: &Path, name: &str) -> bool {
    let data = root.join("plugin-data").join(name);
    http_servers(root, name).is_ok_and(|s| !s.is_empty() && s.iter().all(|(n, _)| crate::mcp_oauth::token_path(&data, n).is_file()))
}

/// One `extensions.com.nousresearch.hermes.config_schema` field.
#[derive(Clone, Debug, PartialEq)]
pub struct Field { pub key: String, pub kind: String, pub description: String, pub default: String, pub secret: bool }
pub fn config_schema(root: &Path, name: &str) -> Vec<Field> {
    let Ok(bytes) = fs::read(dir(root).join(name).join("plugin.json")) else { return vec![] };
    let Ok(manifest) = serde_json::from_slice::<Value>(&bytes) else { return vec![] };
    manifest.pointer("/extensions/com.nousresearch.hermes/config_schema").and_then(Value::as_object).map(|schema| schema.iter().filter_map(|(key, spec)| {
        let spec = spec.as_object()?;
        let kind = spec.get("type").and_then(Value::as_str).unwrap_or("str").to_lowercase();
        let default = match spec.get("default") { Some(Value::String(s)) => s.clone(), Some(Value::Null) | None => String::new(), Some(v) => v.to_string() };
        let secret = spec.get("secret").and_then(Value::as_bool).unwrap_or(false) || ["key", "token", "secret", "password"].iter().any(|w| key.to_lowercase().contains(w));
        Some(Field { key: key.clone(), kind, description: spec.get("description").and_then(Value::as_str).unwrap_or_default().into(), default, secret })
    }).collect()).unwrap_or_default()
}
fn config_path(root: &Path, name: &str) -> PathBuf { root.join("plugin-data").join(name).join("config.json") }
pub fn config(root: &Path, name: &str) -> serde_json::Map<String, Value> {
    fs::read(config_path(root, name)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}
pub fn set_config(root: &Path, name: &str, values: &serde_json::Map<String, Value>) -> Result<()> {
    ensure!(valid_name(name), "That plugin name is not valid");
    let path = config_path(root, name);
    fs::create_dir_all(path.parent().context("No data directory")?)?;
    let temp = tempfile::NamedTempFile::new_in(path.parent().context("No data directory")?)?;
    #[cfg(unix)] { use std::os::unix::fs::PermissionsExt; fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o600))?; }
    serde_json::to_writer_pretty(temp.as_file(), values)?;
    temp.persist(path)?;
    Ok(())
}
/// The pinned commit the catalog now offers differs from the installed one.
pub fn update_available(installed: &Installed, catalog_sha: &str) -> bool { !catalog_sha.is_empty() && installed.sha != catalog_sha }

pub fn uninstall(root: &Path, name: &str) -> Result<()> {
    ensure!(valid_name(name), "That plugin name is not valid");
    let mut list = installed(root)?;
    let before = list.len();
    list.retain(|p| p.name != name);
    ensure!(list.len() < before, "That plugin is not installed");
    crate::skill_catalog::put_documents(root, &format!("plugin/{name}/"), vec![])?;
    let target = dir(root).join(name);
    if target.exists() { fs::remove_dir_all(target)?; }
    save(root, &list)
}

fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if entry.file_name() == ".git" || kind.is_symlink() { continue; }
        let dest = to.join(entry.file_name());
        if kind.is_dir() { copy_tree(&entry.path(), &dest)?; } else if kind.is_file() { fs::copy(entry.path(), dest)?; }
    }
    Ok(())
}

// ---------------------------------------------------------------- package
#[derive(Debug)]
enum Server { Http { url: String, headers: Vec<(String, String)> }, Stdio { command: String, args: Vec<String>, env: Vec<(String, String)>, cwd: Option<String> }, Python }

const BRIDGE: &str = include_str!("../assets/python_plugin_bridge.py");

/// The installed Hermes Python agent: (interpreter, hermes-agent checkout). The `hermes`
/// launcher on PATH names both (`exec <python> -I -c '... sys.path.insert(0, <root>)'`),
/// possibly behind one `exec <launcher> "$@"` hop.
pub fn python_hermes() -> Option<(PathBuf, PathBuf)> {
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from);
    let mut candidates: Vec<PathBuf> = std::env::var_os("PATH").map(|p| std::env::split_paths(&p).map(|d| d.join("hermes")).collect()).unwrap_or_default();
    if let Some(home) = &home { candidates.push(home.join(".local/bin/hermes")); }
    let exec = regex::Regex::new(r#"(?m)^exec (\S+) "\$@""#).ok()?;
    let python = regex::Regex::new(r#"exec (\S*python[0-9.]*) -I -c"#).ok()?;
    let root = regex::Regex::new(r#"sys\.path\.insert\(0, '"'"'(.+?)'"'"'\)"#).ok()?;
    for start in candidates.into_iter().filter(|c| c.is_file()) {
        let mut launcher = start;
        for _ in 0..3 {
            let Ok(text) = fs::read_to_string(&launcher) else { break };
            if let (Some(py), Some(agent)) = (python.captures(&text), root.captures(&text)) {
                let (py, agent) = (PathBuf::from(&py[1]), PathBuf::from(&agent[1]));
                if py.is_file() && agent.join("hermes_cli").is_dir() { return Some((py, agent)); }
                break;
            }
            match exec.captures(&text) { Some(next) => launcher = PathBuf::from(&next[1]), None => break }
        }
    }
    let agent = home?.join(".hermes/hermes-agent");
    [".venv/bin/python", "venv/bin/python", ".venv/Scripts/python.exe"].iter().map(|p| agent.join(p)).find(|p| p.is_file()).map(|py| (py, agent))
}

/// `name:` / `version:` / `description:` from a plugin.yaml (flat scalars only).
fn yaml_field(text: &str, key: &str) -> String {
    text.lines().find_map(|l| l.strip_prefix(key).and_then(|r| r.strip_prefix(':'))).map(|v| v.trim().trim_matches(['"', '\'']).to_owned()).unwrap_or_default()
}
struct Package { name: String, version: String, description: String, skills: Vec<(String, String)>, servers: std::collections::BTreeMap<String, Server> }

fn package_skills(root: &Path) -> Vec<(String, String)> {
    let mut skills = vec![];
    if let Ok(children) = fs::read_dir(root.join("skills")) {
        let mut children: Vec<_> = children.flatten().collect();
        children.sort_by_key(|c| c.file_name());
        for child in children {
            let skill = child.file_name().to_string_lossy().into_owned();
            let path = child.path().join("SKILL.md");
            if child.file_type().is_ok_and(|t| t.is_dir()) && !fs::symlink_metadata(&path).is_ok_and(|m| m.is_symlink())
                && let Ok(doc) = fs::read_to_string(&path) && doc.len() <= 512 * 1024 { skills.push((skill, doc)); }
        }
    }
    skills
}

fn read_package(root: &Path) -> Result<Package> {
    let manifest_path = root.join("plugin.json");
    if !manifest_path.is_file() {
        let yaml = ["plugin.yaml", "plugin.yml"].iter().map(|f| root.join(f)).find(|f| f.is_file());
        let Some(yaml) = yaml else { bail!("This repository has no plugin.json or plugin.yaml at the pinned commit") };
        ensure!(root.join("__init__.py").is_file(), "This Python plugin has no __init__.py");
        let text = fs::read_to_string(&yaml)?;
        let name = yaml_field(&text, "name");
        ensure!(valid_name(&name.replace('_', "-")), "plugin.yaml has no valid name");
        let mut servers = std::collections::BTreeMap::new();
        servers.insert("python".to_owned(), Server::Python);
        return Ok(Package { name, version: yaml_field(&text, "version"), description: yaml_field(&text, "description"), skills: package_skills(root), servers });
    }
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path)?).context("plugin.json is not valid JSON")?;
    if let Some(schema) = manifest["$schema"].as_str() { ensure!(schema == PLUGIN_SCHEMA, "plugin.json declares an unsupported schema"); }
    let name = manifest["name"].as_str().filter(|n| valid_name(n)).context("plugin.json has no valid name")?.to_owned();
    let skills = package_skills(root);
    let mut servers = std::collections::BTreeMap::new();
    if let Ok(bytes) = fs::read(root.join("mcp.json")) {
        let config: Value = serde_json::from_slice(&bytes).context("mcp.json is not valid JSON")?;
        ensure!(config["$schema"].as_str().is_none_or(|s| s == MCP_SCHEMA), "mcp.json declares an unsupported schema");
        for (server, spec) in config["mcpServers"].as_object().context("mcp.json has no mcpServers")? {
            let strings = |v: &Value| v.as_object().map(|m| m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_owned()))).collect()).unwrap_or_default();
            let parsed = match spec["type"].as_str() {
                Some("streamable-http") => Server::Http { url: spec["url"].as_str().context("A server has no url")?.to_owned(), headers: strings(&spec["headers"]) },
                Some("stdio") => {
                    let command = spec["command"].as_str().filter(|c| !c.is_empty() && !c.contains('\0')).context("A stdio server has no command")?.to_owned();
                    ensure!(command.starts_with("./") || !(command.contains('/') || command.contains('\\') || command.contains(char::is_whitespace)), "A stdio command must be a bare executable or begin with ./");
                    let env: Vec<(String, String)> = strings(&spec["env"]);
                    ensure!(!env.iter().any(|(k, _)| k == "PLUGIN_ROOT" || k == "PLUGIN_DATA"), "PLUGIN_ROOT and PLUGIN_DATA are reserved");
                    Server::Stdio { command, args: spec["args"].as_array().map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect()).unwrap_or_default(),
                        env, cwd: spec["cwd"].as_str().map(str::to_owned) }
                }
                _ => continue, // sse and unknown types are not portable
            };
            servers.insert(server.clone(), parsed);
        }
    }
    Ok(Package { name, version: manifest["version"].as_str().unwrap_or_default().to_owned(), description: manifest["description"].as_str().unwrap_or_default().to_owned(), skills, servers })
}

fn prefix(server: &str) -> String {
    let clean: String = server.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '_' }).take(20).collect();
    format!("mcp_{clean}")
}

/// Connect every enabled plugin's servers. A server that fails is reported, not fatal.
pub(crate) async fn connect_all(root: &Path) -> (Vec<Mcp>, Vec<String>) {
    let (mut live, mut errors) = (vec![], vec![]);
    for plugin in installed(root).unwrap_or_default().into_iter().filter(|p| p.enabled && !p.servers.is_empty()) {
        let plugin_root = dir(root).join(&plugin.name);
        let data = root.join("plugin-data").join(&plugin.name);
        let package = match read_package(&plugin_root) { Ok(p) => p, Err(e) => { errors.push(format!("{}: {e:#}", plugin.title)); continue } };
        let values = config(root, &plugin.name);
        let defaults: Vec<Field> = config_schema(root, &plugin.name);
        let setting = |key: &str| values.get(key).map(|v| v.as_str().map_or_else(|| v.to_string(), str::to_owned)).or_else(|| defaults.iter().find(|f| f.key == key).map(|f| f.default.clone())).unwrap_or_default();
        let expand = |v: &str| {
            let mut out = v.replace("${PLUGIN_ROOT}", &plugin_root.to_string_lossy()).replace("${PLUGIN_DATA}", &data.to_string_lossy());
            for field in &defaults { out = out.replace(&format!("${{config.{}}}", field.key), &setting(&field.key)); }
            out
        };
        let config_env: Vec<(String, String)> = defaults.iter().map(|f| (format!("PLUGIN_CONFIG_{}", f.key.to_uppercase().replace(|c: char| !c.is_ascii_alphanumeric(), "_")), setting(&f.key))).collect();
        for (name, server) in package.servers {
            let connected = match server {
                Server::Http { url, headers } => {
                    let (url, headers) = (expand(&url), headers.into_iter().map(|(k, v)| (k, expand(&v))).collect());
                    let token = crate::mcp_oauth::token(&data, &name).await.ok().flatten();
                    let result = Mcp::connect_http(&url, token, headers, &prefix(&name)).await;
                    if let Err(e) = &result && format!("{e:#}").contains("401") { errors.push(format!("{} › {name}: needs sign-in (Capabilities › Plugins › Sign in)", plugin.title)); continue }
                    result
                }
                Server::Python => {
                    let Some((python, agent)) = python_hermes() else { errors.push(format!("{}: needs the Hermes Python agent installed", plugin.title)); continue };
                    let bridge = data.join("python_plugin_bridge.py");
                    if fs::read_to_string(&bridge).ok().as_deref() != Some(BRIDGE) && let Err(e) = fs::create_dir_all(&data).and_then(|_| fs::write(&bridge, BRIDGE)) { errors.push(format!("{}: {e}", plugin.title)); continue }
                    let args = vec!["-I".to_owned(), bridge.to_string_lossy().into_owned(), agent.to_string_lossy().into_owned(), plugin_root.to_string_lossy().into_owned()];
                    Mcp::connect_stdio(Stdio { command: python.to_string_lossy().into_owned(), args, env: config_env.clone(), cwd: plugin_root.clone() }, &prefix(&plugin.name)).await
                }
                Server::Stdio { command, args, mut env, cwd } => {
                    let command = match command.strip_prefix("./") { Some(rel) => match inside(&plugin_root, rel) { Some(p) => p.to_string_lossy().into_owned(), None => { errors.push(format!("{name}: bad command path")); continue } }, None => command };
                    let cwd = match cwd { Some(c) => PathBuf::from(expand(&c)), None => plugin_root.clone() };
                    let scoped = cwd.canonicalize().ok().is_some_and(|c| c.starts_with(plugin_root.canonicalize().unwrap_or_default()) || c.starts_with(data.canonicalize().unwrap_or_default()));
                    if !scoped { errors.push(format!("{name}: cwd must stay inside the plugin")); continue }
                    env.iter_mut().for_each(|(_, v)| *v = expand(v));
                    env.extend(config_env.iter().cloned());
                    env.push(("PLUGIN_ROOT".into(), plugin_root.to_string_lossy().into_owned()));
                    env.push(("PLUGIN_DATA".into(), data.to_string_lossy().into_owned()));
                    Mcp::connect_stdio(Stdio { command, args: args.iter().map(|a| expand(a)).collect(), env, cwd }, &prefix(&name)).await
                }
            };
            match connected { Ok(mcp) => live.push(mcp), Err(e) => errors.push(format!("{} › {name}: {e:#}", plugin.title)) }
        }
    }
    (live, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reads_agent_plugin_packages_and_python_ones() {
        let dir = tempfile::tempdir().unwrap();
        let pkg = dir.path().join("pkg");
        fs::create_dir_all(pkg.join("skills/safe-tx")).unwrap();
        fs::write(pkg.join("plugin.json"), format!(r#"{{"$schema":"{PLUGIN_SCHEMA}","name":"aave-mcp","version":"1.0.0","description":"Aave"}}"#)).unwrap();
        fs::write(pkg.join("skills/safe-tx/SKILL.md"), "---\nname: safe-tx\ndescription: Safe transactions\n---\nBody").unwrap();
        fs::write(pkg.join("mcp.json"), format!(r#"{{"$schema":"{MCP_SCHEMA}","mcpServers":{{"aave":{{"type":"streamable-http","url":"https://mcp.aave.com"}},"local":{{"type":"stdio","command":"./bin/server","args":["${{PLUGIN_DATA}}"]}},"old":{{"type":"sse","url":"https://x"}}}}}}"#)).unwrap();
        let package = read_package(&pkg).unwrap();
        assert_eq!(package.name, "aave-mcp");
        assert_eq!(package.skills[0].0, "safe-tx");
        assert_eq!(package.servers.keys().collect::<Vec<_>>(), ["aave", "local"], "sse is not portable");
        // Commands must be bare or package-relative.
        fs::write(pkg.join("mcp.json"), r#"{"mcpServers":{"x":{"type":"stdio","command":"/bin/sh"}}}"#).unwrap();
        assert!(read_package(&pkg).is_err());
        let py = dir.path().join("py");
        fs::create_dir_all(&py).unwrap();
        fs::write(py.join("plugin.yaml"), "name: hello_py\nversion: 0.2.0\n").unwrap();
        assert!(format!("{:#}", read_package(&py).err().unwrap()).contains("__init__.py"));
        fs::write(py.join("__init__.py"), "def register(ctx): pass\n").unwrap();
        let package = read_package(&py).unwrap();
        assert_eq!((package.name.as_str(), package.version.as_str()), ("hello_py", "0.2.0"));
        assert_eq!(package.servers.keys().collect::<Vec<_>>(), ["python"]);
        assert!(inside(&pkg, "../etc").is_none() && inside(&pkg, "/etc").is_none() && inside(&pkg, "plugins/mcp").is_some());
        assert!(!valid_name("../x") && valid_name("aave-mcp"));
    }
    #[tokio::test]
    async fn an_enabled_plugin_runs_its_stdio_server_and_a_disabled_one_does_not() {
        let root = tempfile::tempdir().unwrap();
        let pkg = root.path().join("plugins/echo-kit");
        fs::create_dir_all(&pkg).unwrap();
        fs::write(pkg.join("plugin.json"), r#"{"name":"echo-kit","version":"0.1.0"}"#).unwrap();
        fs::write(pkg.join("mcp.json"), r#"{"mcpServers":{"echo":{"type":"stdio","command":"python3","args":["${PLUGIN_ROOT}/server.py"]}}}"#).unwrap();
        fs::write(pkg.join("server.py"), r#"
import json, sys, os
for line in sys.stdin:
    m = json.loads(line)
    if "id" not in m: continue
    r = {"initialize": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}},
         "tools/list": {"tools": [{"name": "echo", "description": "Echo", "inputSchema": {"type": "object"}}]},
         "tools/call": {"content": [{"type": "text", "text": m.get("params", {}).get("arguments", {}).get("v", "") + "|" + os.path.basename(os.environ["PLUGIN_DATA"])}]}}[m["method"]]
    print(json.dumps({"jsonrpc": "2.0", "id": m["id"], "result": r}), flush=True)
"#).unwrap();
        fs::create_dir_all(root.path().join("plugin-data/echo-kit")).unwrap();
        let mut record = Installed { name: "echo-kit".into(), title: "Echo".into(), enabled: true, servers: vec!["echo".into()], ..Default::default() };
        save(root.path(), std::slice::from_ref(&record)).unwrap();
        let (mut live, errors) = connect_all(root.path()).await;
        assert!(errors.is_empty(), "{errors:?}");
        assert!(live[0].contains("mcp_echo__echo"));
        let reply = live[0].call("mcp_echo__echo", &serde_json::json!({"v": "hi"})).await.unwrap();
        assert_eq!(reply["content"][0]["text"], "hi|echo-kit");
        live[0].close().await;
        record.enabled = false;
        save(root.path(), &[record]).unwrap();
        assert!(connect_all(root.path()).await.0.is_empty());
    }
    /// Needs the Hermes Python agent on this machine: `cargo test -- --ignored python_plugin`
    #[tokio::test] #[ignore]
    async fn a_python_plugin_runs_through_the_installed_hermes() {
        assert!(python_hermes().is_some(), "no Hermes Python agent found");
        let root = tempfile::tempdir().unwrap();
        let pkg = root.path().join("plugins/shout");
        fs::create_dir_all(&pkg).unwrap();
        fs::write(pkg.join("plugin.yaml"), "name: shout\nversion: 1.0.0\ndescription: Shouts\n").unwrap();
        fs::write(pkg.join("__init__.py"), r#"
import json

def _shout(args, **_):
    return json.dumps({"text": str(args.get("text", "")).upper() + "!"})

def register(ctx):
    ctx.register_tool("shout_text", "shout", {"name": "shout_text", "description": "Shout text",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}, _shout)
"#).unwrap();
        save(root.path(), &[Installed { name: "shout".into(), title: "Shout".into(), enabled: true, servers: vec!["python".into()], ..Default::default() }]).unwrap();
        let (mut live, errors) = connect_all(root.path()).await;
        assert!(errors.is_empty(), "{errors:?}");
        assert!(live[0].contains("mcp_shout__shout_text"), "{:?}", live[0].specs());
        let reply = live[0].call("mcp_shout__shout_text", &serde_json::json!({"text": "hi"})).await.unwrap();
        assert!(reply["content"][0]["text"].as_str().unwrap().contains("HI!"), "{reply}");
        live[0].close().await;
    }
    /// Network: installs the real catalog entry at its pin. `cargo test -- --ignored live_install`
    #[test] #[ignore]
    fn live_install_of_a_catalog_agent_plugin() {
        let root = tempfile::tempdir().unwrap();
        let entry = CatalogEntry { name: "aave-mcp".into(), title: "Aave".into(), repo: "https://github.com/aave/skills".into(), sha: "7132f6e952f9100f16577c33774a1e63948f7fab".into(), subdir: "plugins/mcp".into() };
        let record = install(root.path(), &entry).unwrap();
        eprintln!("{record:?}");
        assert!(!record.servers.is_empty() && !record.skills.is_empty());
        assert!(crate::skill_catalog::status(root.path()).unwrap().installed.iter().any(|id| id.starts_with("plugin/aave-mcp/")));
        let (live, errors) = tokio::runtime::Runtime::new().unwrap().block_on(connect_all(root.path()));
        eprintln!("servers={} errors={errors:?} tools={:?}", live.len(), live.iter().flat_map(|m| m.specs()).filter_map(|s| s["function"]["name"].as_str().map(str::to_owned)).take(6).collect::<Vec<_>>());
        assert_eq!(live.len(), 1, "{errors:?}");
        uninstall(root.path(), "aave-mcp").unwrap();
        assert!(installed(root.path()).unwrap().is_empty() && !root.path().join("plugins/aave-mcp").exists());
    }
}
