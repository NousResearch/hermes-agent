//! Profiles (app/profiles): each one is its own Hermes — own keys, memory,
//! sessions and settings — living at `<root>/profiles/<name>`. The default
//! profile is the library root itself. Bots are profiles too (Bot Mode), so
//! they are listed, but renamed and deleted from the Bots page.
use crate::Library;
use anyhow::{Context, Result, bail, ensure};
use std::{fs, path::{Path, PathBuf}, process::Command};

#[derive(Clone, Debug, PartialEq)]
pub struct Profile { pub name: String, pub path: PathBuf, pub is_default: bool, pub is_bot: bool, pub model: String, pub skill_count: usize, pub sessions: usize }

pub fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 64 && name.bytes().next().is_some_and(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_') && name != "default"
}

fn describe(name: &str, path: &Path, is_default: bool, is_bot: bool) -> Profile {
    let prefs: serde_json::Value = fs::read(path.join("desktop-preferences.json")).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let skill_count = hermes_engine::skill_catalog::status(path).map(|s| s.installed.difference(&s.disabled).count()).unwrap_or(0);
    let sessions = fs::read(path.join("library.json")).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v["sessions"].as_array().map(Vec::len)).unwrap_or(0);
    Profile { name: name.into(), path: path.into(), is_default, is_bot, model: prefs["model"].as_str().unwrap_or_default().into(), skill_count, sessions }
}

impl Library {
    fn profiles_dir(&self) -> PathBuf { self.root().join("profiles") }
    pub fn profiles(&self) -> Result<Vec<Profile>> {
        let bots: Vec<String> = self.bots().unwrap_or_default().into_iter().map(|b| b.name).collect();
        let mut out = vec![describe("default", self.root(), true, false)];
        if let Ok(entries) = fs::read_dir(self.profiles_dir()) {
            let mut named: Vec<_> = entries.flatten().filter(|e| e.file_type().is_ok_and(|t| t.is_dir())).filter_map(|e| e.file_name().into_string().ok()).filter(|n| valid_name(n)).collect();
            named.sort();
            for name in named { let is_bot = bots.contains(&name); out.push(describe(&name, &self.profiles_dir().join(&name), false, is_bot)); }
        }
        Ok(out)
    }
    pub fn profile_path(&self, name: &str) -> Result<PathBuf> {
        if name == "default" { return Ok(self.root().to_owned()); }
        ensure!(valid_name(name), "Lowercase letters, digits, hyphens and underscores. Must start with a letter or digit.");
        Ok(self.profiles_dir().join(name))
    }
    pub fn create_profile(&self, name: &str) -> Result<Profile> {
        let path = self.profile_path(name)?;
        ensure!(name != "default" && !path.exists(), "A profile with that name already exists");
        fs::create_dir_all(&path)?;
        Ok(describe(name, &path, false, false))
    }
    pub fn rename_profile(&self, from: &str, to: &str) -> Result<()> {
        ensure!(from != "default", "The default profile can't be renamed");
        ensure!(!self.bots()?.iter().any(|b| b.name == from), "Rename bots from the Bots page");
        let (a, b) = (self.profile_path(from)?, self.profile_path(to)?);
        ensure!(a.is_dir(), "No such profile");
        ensure!(!b.exists(), "A profile with that name already exists");
        fs::rename(a, b)?;
        Ok(())
    }
    pub fn delete_profile(&self, name: &str) -> Result<()> {
        ensure!(name != "default", "The default profile can't be deleted");
        ensure!(!self.bots()?.iter().any(|b| b.name == name), "Delete bots from the Bots page");
        let path = self.profile_path(name)?;
        ensure!(path.is_dir(), "No such profile");
        fs::remove_dir_all(path)?;
        Ok(())
    }
    pub fn profile_soul(&self, name: &str) -> Result<String> { Ok(fs::read_to_string(self.profile_path(name)?.join("SOUL.md")).unwrap_or_default()) }
    pub fn set_profile_soul(&self, name: &str, text: &str) -> Result<()> {
        let path = self.profile_path(name)?;
        fs::create_dir_all(&path)?;
        let file = tempfile::NamedTempFile::new_in(&path)?;
        fs::write(file.path(), text)?;
        file.persist(path.join("SOUL.md"))?;
        Ok(())
    }
    /// A `.tar.gz` of one named profile (its sessions, memory, settings, keys).
    pub fn export_profile(&self, name: &str, dest: &Path) -> Result<()> {
        ensure!(name != "default", "Back up the default profile from Settings › Backup");
        ensure!(self.profile_path(name)?.is_dir(), "No such profile");
        let status = Command::new("tar").arg("-czf").arg(dest).arg("-C").arg(self.profiles_dir()).arg(name).status().context("tar is not available")?;
        ensure!(status.success(), "Could not write the profile archive");
        Ok(())
    }
    /// Import an archive made by `export_profile`; returns the profile's name.
    pub fn import_profile(&self, archive: &Path) -> Result<String> {
        let listing = Command::new("tar").arg("-tzf").arg(archive).output().context("tar is not available")?;
        ensure!(listing.status.success(), "That file is not a profile archive");
        let entries = String::from_utf8_lossy(&listing.stdout).into_owned();
        let mut tops = entries.lines().filter_map(|l| l.trim_start_matches("./").split('/').next()).filter(|t| !t.is_empty()).collect::<Vec<_>>();
        tops.sort(); tops.dedup();
        let [name] = tops.as_slice() else { bail!("A profile archive holds exactly one profile folder") };
        ensure!(valid_name(name), "The archive's profile name is not valid");
        ensure!(!entries.lines().any(|l| l.starts_with('/') || l.split('/').any(|p| p == "..")), "The archive has unsafe paths");
        ensure!(!self.profiles_dir().join(name).exists(), "A profile named {name} already exists");
        fs::create_dir_all(self.profiles_dir())?;
        let status = Command::new("tar").arg("-xzf").arg(archive).arg("-C").arg(self.profiles_dir()).status()?;
        ensure!(status.success(), "Could not unpack the profile archive");
        Ok(name.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn create_rename_soul_export_import_delete() {
        let dir = tempfile::tempdir().unwrap();
        let library = Library::open(dir.path()).unwrap();
        assert_eq!(library.profiles().unwrap().len(), 1, "just the default");
        library.create_profile("work").unwrap();
        assert!(library.create_profile("work").is_err() && library.create_profile("Bad Name").is_err() && library.create_profile("default").is_err());
        library.set_profile_soul("work", "You are focused.").unwrap();
        library.rename_profile("work", "office").unwrap();
        assert_eq!(library.profile_soul("office").unwrap(), "You are focused.");
        let archive = dir.path().join("office.tar.gz");
        library.export_profile("office", &archive).unwrap();
        library.delete_profile("office").unwrap();
        assert_eq!(library.import_profile(&archive).unwrap(), "office");
        assert!(library.import_profile(&archive).is_err(), "no overwrite");
        let names: Vec<_> = library.profiles().unwrap().into_iter().map(|p| p.name).collect();
        assert_eq!(names, ["default", "office"]);
        assert!(library.delete_profile("default").is_err());
    }
}
