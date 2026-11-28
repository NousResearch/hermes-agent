//! Private named endpoint credentials. A credential is only returned to its
//! exact canonical endpoint scope, never copied when a URL or host changes.
use crate::Config;
use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointProfile {
    pub id: String,
    pub name: String,
    pub base_url: String,
    pub model: String,
    pub has_key: bool,
}
#[derive(Serialize, Deserialize)]
struct Record {
    id: String,
    name: String,
    base_url: String,
    model: String,
    scope: String,
    api_key: String,
}
#[derive(Serialize, Deserialize, Default)]
struct Document {
    #[serde(default)]
    endpoints: Vec<Record>,
}
#[derive(Clone)]
pub struct EndpointStore {
    dir: PathBuf,
}

pub fn scope(base: &str) -> Result<String> {
    let mut config = Config {
        base_url: base.into(),
        model: "scope".into(),
        api_key: "scope-validation".into(),
    };
    config.validate()?;
    let mut url = Url::parse(config.base_url.trim())?;
    let path = url.path().trim_end_matches('/').to_owned();
    url.set_path(&path);
    config.api_key.clear();
    Ok(url.to_string().trim_end_matches('/').to_owned())
}
impl EndpointStore {
    pub fn open(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)?;
        ensure!(
            !fs::symlink_metadata(dir)?.file_type().is_symlink(),
            "Provider credential directory must not be a link"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(dir, fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self {
            dir: dir.canonicalize()?,
        })
    }
    pub fn list(&self) -> Result<Vec<EndpointProfile>> {
        let _guard = self.lock()?;
        Ok(self.read()?.endpoints.iter().map(public).collect())
    }
    pub fn config(&self, id: &str) -> Result<Config> {
        let _guard = self.lock()?;
        let doc = self.read()?;
        let record = doc
            .endpoints
            .iter()
            .find(|record| record.id == id)
            .context("Saved endpoint no longer exists")?;
        ensure!(
            scope(&record.base_url)? == record.scope,
            "Saved credential scope does not match its endpoint"
        );
        Ok(Config {
            base_url: record.base_url.clone(),
            model: record.model.clone(),
            api_key: record.api_key.clone(),
        })
    }
    pub fn credential(&self, base: &str) -> Result<Option<String>> {
        let wanted = scope(base)?;
        let _guard = self.lock()?;
        Ok(self
            .read()?
            .endpoints
            .into_iter()
            .find(|record| {
                record.scope == wanted && scope(&record.base_url).is_ok_and(|s| s == wanted)
            })
            .map(|record| record.api_key))
    }
    pub fn save(&self, id: Option<&str>, name: &str, config: &Config) -> Result<EndpointProfile> {
        config.validate()?;
        ensure!(
            !crate::portal::is_portal(&config.base_url),
            "Nous OAuth tokens are managed by account sign-in, not saved as endpoint API keys"
        );
        ensure!(
            !name.trim().is_empty() && name.len() <= 128,
            "Enter an endpoint name up to 128 characters"
        );
        let scope = scope(&config.base_url)?;
        let _guard = self.lock()?;
        let mut doc = self.read()?;
        let id = id.map(str::to_owned).unwrap_or_else(|| {
            format!(
                "endpoint-{}",
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
            )
        });
        let record = Record {
            id: id.clone(),
            name: name.trim().into(),
            base_url: scope.clone(),
            model: config.model.trim().into(),
            scope,
            api_key: config.api_key.trim().into(),
        };
        let result = public(&record);
        if let Some(index) = doc.endpoints.iter().position(|existing| existing.id == id) {
            doc.endpoints[index] = record;
        } else {
            ensure!(doc.endpoints.len() < 64, "Saved endpoint limit reached");
            doc.endpoints.push(record);
        }
        self.write(&doc)?;
        Ok(result)
    }
    pub fn remove(&self, id: &str) -> Result<()> {
        let _guard = self.lock()?;
        let mut doc = self.read()?;
        doc.endpoints.retain(|record| record.id != id);
        self.write(&doc)
    }
    fn lock(&self) -> Result<File> {
        let path = self.dir.join("endpoints.lock");
        reject_link(&path)?;
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true).truncate(false);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let file = options.open(path)?;
        file.try_lock_exclusive()
            .context("Provider credentials are busy in another window")?;
        Ok(file)
    }
    fn read(&self) -> Result<Document> {
        let path = self.dir.join("endpoints.json");
        reject_link(&path)?;
        match File::open(path) {
            Ok(file) => {
                ensure!(
                    file.metadata()?.len() <= 1_048_576,
                    "Provider credential file is too large"
                );
                serde_json::from_reader(file).context(
                    "Saved provider credentials could not be read; the file was left unchanged",
                )
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Document::default()),
            Err(error) => Err(error.into()),
        }
    }
    fn write(&self, doc: &Document) -> Result<()> {
        let path = self.dir.join("endpoints.json");
        reject_link(&path)?;
        let mut file = NamedTempFile::new_in(&self.dir)?;
        serde_json::to_writer(&mut file, doc)?;
        file.flush()?;
        file.as_file().sync_all()?;
        file.persist(path)?;
        Ok(())
    }
}
fn public(record: &Record) -> EndpointProfile {
    EndpointProfile {
        id: record.id.clone(),
        name: record.name.clone(),
        base_url: record.base_url.clone(),
        model: record.model.clone(),
        has_key: !record.api_key.is_empty(),
    }
}
fn reject_link(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => ensure!(
            metadata.is_file() && !metadata.file_type().is_symlink(),
            "Provider storage path must be a regular file"
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    Ok(())
}
