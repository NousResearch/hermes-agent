//! Session-owned snapshots of files explicitly attached in the desktop composer.
//! History stores references. Provider requests read only validated local blobs.
use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use ring::digest::{SHA256, digest};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;

pub const MAX_ATTACHMENTS: usize = 8;
const MAX_IMAGE: u64 = 4 * 1024 * 1024;
const MAX_TEXT: usize = 128 * 1024;
const MAX_TOTAL: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct Attachment {
    pub name: String,
    digest: String,
    pub media_type: String,
    bytes: usize,
}

fn hash(data: &[u8]) -> String {
    use std::fmt::Write;
    let mut hex = String::with_capacity(64);
    for byte in digest(&SHA256, data).as_ref() {
        write!(hex, "{byte:02x}").unwrap();
    }
    hex
}

fn media_type(data: &[u8]) -> Result<&'static str> {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return Ok("image/png");
    }
    if data.starts_with(&[0xff, 0xd8, 0xff]) {
        return Ok("image/jpeg");
    }
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        return Ok("image/gif");
    }
    if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
        return Ok("image/webp");
    }
    ensure!(
        !data.contains(&0) && std::str::from_utf8(data).is_ok(),
        "Choose a UTF-8 text file, PNG, JPEG, GIF or WebP image"
    );
    ensure!(
        data.len() <= MAX_TEXT,
        "Text attachments must be 128 KiB or smaller"
    );
    Ok("text/plain")
}
fn blob_dir(session: &Path) -> Result<PathBuf> {
    let path = session.join("attachments");
    if let Err(error) = fs::create_dir(&path) {
        if error.kind() != std::io::ErrorKind::AlreadyExists {
            return Err(error.into());
        }
    }
    ensure!(
        fs::symlink_metadata(&path)?.file_type().is_dir(),
        "Invalid attachment directory"
    );
    Ok(path)
}

impl Attachment {
    pub fn capture(session: &Path, selected: &Path) -> Result<Self> {
        ensure!(fs::metadata(selected)?.is_file(), "Select a regular file");
        let file = fs::File::open(selected).context("Could not read the selected file")?;
        let meta = file.metadata()?;
        ensure!(
            meta.is_file() && meta.len() <= MAX_IMAGE,
            "Attachments must be regular files of 4 MiB or smaller"
        );
        let name = selected
            .file_name()
            .and_then(|value| value.to_str())
            .context("The filename must be valid UTF-8")?
            .to_owned();
        ensure!(
            !name.chars().any(char::is_control) && name.len() <= 1024,
            "Unsupported filename"
        );
        let mut data = Vec::new();
        file.take(MAX_IMAGE + 1).read_to_end(&mut data)?;
        ensure!(
            data.len() as u64 <= MAX_IMAGE,
            "Attachment grew past 4 MiB while reading"
        );
        let media_type = media_type(&data)?.to_owned();
        let key = hash(&data);
        let dir = blob_dir(session)?;
        let mut file = NamedTempFile::new_in(&dir)?;
        file.write_all(&data)?;
        file.as_file().sync_all()?;
        match file.persist_noclobber(dir.join(&key)) {
            Ok(_) => {}
            Err(e) if e.error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.error.into()),
        }
        Ok(Self {
            name,
            digest: key,
            media_type,
            bytes: data.len(),
        })
    }

    fn data(&self, session: &Path) -> Result<Vec<u8>> {
        ensure!(
            self.digest.len() == 64 && self.digest.bytes().all(|b| b.is_ascii_hexdigit()),
            "Invalid attachment reference"
        );
        let path = blob_dir(session)?.join(&self.digest);
        let meta = fs::symlink_metadata(&path).context("An attached file snapshot is missing")?;
        ensure!(
            meta.file_type().is_file()
                && meta.len() <= MAX_IMAGE
                && meta.len() == self.bytes as u64,
            "Invalid attachment snapshot"
        );
        let mut data = Vec::new();
        fs::File::open(&path)?
            .take(MAX_IMAGE + 1)
            .read_to_end(&mut data)?;
        ensure!(
            hash(&data) == self.digest && media_type(&data)? == self.media_type,
            "The saved attachment has changed"
        );
        Ok(data)
    }
    pub fn validate_set(files: &[Self]) -> Result<()> {
        ensure!(
            files.len() <= MAX_ATTACHMENTS,
            "Attach up to eight files per message"
        );
        let total = files
            .iter()
            .try_fold(0usize, |sum, f| sum.checked_add(f.bytes))
            .context("Invalid attachment sizes")?;
        ensure!(
            total <= MAX_TOTAL,
            "Combined attachments must be 8 MiB or smaller"
        );
        Ok(())
    }
    pub(crate) fn user_content(session: &Path, prompt: &str, files: &[Self]) -> Result<Value> {
        if files.is_empty() {
            return Ok(json!(prompt));
        }
        Self::validate_set(files)?;
        let mut content = Vec::new();
        if !prompt.is_empty() {
            content.push(json!({"type":"text","text":prompt}));
        }
        for file in files {
            let data = file.data(session)?;
            if file.media_type == "text/plain" {
                let label = serde_json::to_string(&file.name)?;
                let source = std::str::from_utf8(&data)?;
                content
                    .push(json!({"type":"text","text":format!("Attached file {label}\n{source}")}));
            } else {
                content.push(json!({"type":"image_url","image_url":{"url":format!("data:{};base64,{}", file.media_type, STANDARD.encode(data))}}));
            }
        }
        Ok(Value::Array(content))
    }
}

pub(crate) fn anthropic_user_content(content: &Value) -> Result<Vec<Value>> {
    if let Some(text) = content.as_str() {
        return Ok(vec![json!({"type":"text","text":text})]);
    }
    content.as_array().context("Invalid user content")?.iter().map(|p| match p["type"].as_str() {
        Some("text") => Ok(p.clone()),
        Some("image_url") => {
            let (media,data)=p["image_url"]["url"].as_str().and_then(|s|s.strip_prefix("data:")).and_then(|s|s.split_once(";base64,")).context("Invalid image attachment")?;
            Ok(json!({"type":"image","source":{"type":"base64","media_type":media,"data":data}}))
        }
        _=>bail!("Unsupported attachment content"),
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snapshots_ignore_later_changes_and_reject_tampering() {
        let dir = tempfile::tempdir().unwrap();
        let original = dir.path().join("example.txt");
        fs::write(&original, "captured").unwrap();
        let a = Attachment::capture(dir.path(), &original).unwrap();
        fs::write(&original, "replacement").unwrap();
        let content =
            Attachment::user_content(dir.path(), "Read", std::slice::from_ref(&a)).unwrap();
        assert_eq!(
            content[1]["text"],
            "Attached file \"example.txt\"\ncaptured"
        );
        fs::write(dir.path().join("attachments").join(&a.digest), "modified").unwrap();
        assert!(a.data(dir.path()).is_err());
    }
    #[test]
    fn image_is_sent_in_the_selected_providers_wire_format() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("image");
        fs::write(&file, b"\x89PNG\r\n\x1a\nfixture").unwrap();
        let attachment = Attachment::capture(dir.path(), &file).unwrap();
        let value = Attachment::user_content(dir.path(), "describe", &[attachment]).unwrap();
        assert!(
            value[1]["image_url"]["url"]
                .as_str()
                .unwrap()
                .starts_with("data:image/png;base64,")
        );
        let blocks = anthropic_user_content(&value).unwrap();
        assert_eq!(blocks[1]["source"]["media_type"], "image/png");
    }
}
