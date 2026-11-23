//! Native development vault. Field names and validation follow official Hermes
//! 0284ce17, app/settings/vault-settings.tsx and i18n/en.ts. Only intentional
//! add/edit/delete actions touch exact named OS credential items; listing reads
//! this library's metadata index. Agent consumption, browser filling, OTP code
//! generation and third-party password managers are not connected to this vault.

use crate::appearance::text;
use crate::{Hermes, Message as AppMessage, ui};
use chrono::Utc;
use fs2::FileExt;
use iced::{
    Alignment, Border, Element, Fill, Task,
    widget::{Space, button, column, container, pick_list, row, rule, text_input},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fmt,
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tempfile::NamedTempFile;
use uuid::Uuid;
use zeroize::{Zeroize, Zeroizing};

const INDEX_FILE: &str = "desktop-vault-index.json";
const LOCK_FILE: &str = "desktop-vault.lock";
const MAX_INDEX_BYTES: u64 = 1_048_576;
#[cfg(target_os = "windows")]
const MAX_SECRET_BYTES: usize = 2_560;
#[cfg(not(target_os = "windows"))]
const MAX_SECRET_BYTES: usize = 65_536;
// Deliberately distinct from stock Hermes and every provider/account key store.
const KEYCHAIN_SERVICE: &str = "org.nousresearch.hermes-rust.development.credential-vault.v1";
#[cfg(target_os = "macos")]
const STORE_NAME: &str = "macOS Keychain";
#[cfg(target_os = "windows")]
const STORE_NAME: &str = "Windows Credential Manager";
#[cfg(target_os = "linux")]
const STORE_NAME: &str = "Linux Secret Service";
#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
const STORE_NAME: &str = "the operating system credential store";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    #[default]
    Login,
    Payment,
    Address,
}
impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Login => "Login",
            Self::Payment => "Payment card",
            Self::Address => "Address",
        })
    }
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IdentifierType {
    #[default]
    Email,
    Phone,
    Username,
}
impl IdentifierType {
    fn key(self) -> &'static str {
        match self {
            Self::Email => "email",
            Self::Phone => "phone",
            Self::Username => "username",
        }
    }
}
impl fmt::Display for IdentifierType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Email => "Email",
            Self::Phone => "Phone",
            Self::Username => "Username",
        })
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Field {
    Label,
    Origin,
    Identifier,
    Password,
    Otp,
    CardNumber,
    CardName,
    ExpMonth,
    ExpYear,
    Cvc,
    Postal,
    Line1,
    Line2,
    City,
    State,
    Country,
}
impl Field {
    fn key(self) -> &'static str {
        match self {
            Self::Label => "label",
            Self::Origin => "origin",
            Self::Identifier => "identifier",
            Self::Password => "password",
            Self::Otp => "otp_secret",
            Self::CardNumber => "card_number",
            Self::CardName => "cardholder_name",
            Self::ExpMonth => "exp_month",
            Self::ExpYear => "exp_year",
            Self::Cvc => "cvc",
            Self::Postal => "postal_code",
            Self::Line1 => "address_line1",
            Self::Line2 => "address_line2",
            Self::City => "city",
            Self::State => "state",
            Self::Country => "country",
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Item {
    id: String,
    kind: Kind,
    label: String,
    origin: String,
    created_at: String,
    updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    identifier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    identifier_type: Option<String>,
    #[serde(default)]
    has_otp: bool,
}

/// This type contains metadata only. Passwords, OTP seeds, payment and address
/// fields are never serialized into the index or sent to the model/transcript.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Index {
    version: u8,
    namespace: String,
    items: Vec<Item>,
}
impl Index {
    fn empty() -> Self {
        Self {
            version: 1,
            namespace: new_id(),
            items: Vec::new(),
        }
    }
    fn validate(&self) -> Result<(), String> {
        if self.version != 1 || !valid_id(&self.namespace) || self.items.len() > 2_000 {
            return Err("This vault index has an unsupported format.".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for item in &self.items {
            if !valid_id(&item.id)
                || !seen.insert(&item.id)
                || item.label.len() > 1_024
                || item.origin.len() > 4_096
                || item.identifier.as_ref().is_some_and(|s| s.len() > 8_192)
            {
                return Err("This vault index contains invalid item metadata.".into());
            }
        }
        Ok(())
    }
}

#[derive(Clone, Default)]
pub(crate) struct Form {
    kind: Kind,
    identifier_type: IdentifierType,
    fields: BTreeMap<Field, String>,
}
impl Drop for Form {
    fn drop(&mut self) {
        for value in self.fields.values_mut() {
            value.zeroize();
        }
    }
}
impl Form {
    fn value(&self, field: Field) -> &str {
        self.fields
            .get(&field)
            .map(String::as_str)
            .unwrap_or_default()
    }
    fn set(&mut self, field: Field, mut value: String) {
        // Keep a pasted secret bounded without ever reporting its contents.
        if let Some((end, _)) = value.char_indices().nth(8_192) {
            value.truncate(end);
        }
        if let Some(mut old) = self.fields.insert(field, value) {
            old.zeroize();
        }
    }
    fn validate(&self) -> Result<(), String> {
        if self.value(Field::Label).trim().is_empty() {
            return Err("A label is required.".into());
        }
        if self.value(Field::Label).len() > 1_024 {
            return Err("Use a label under 1,024 bytes.".into());
        }
        let origin = self.value(Field::Origin).trim();
        let valid = url::Url::parse(origin).is_ok_and(|url| {
            matches!(url.scheme(), "http" | "https")
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
        });
        if !valid || origin.len() > 4_096 {
            return Err(
                "Enter a valid URL like https://example.com. Do not include a password in the URL."
                    .into(),
            );
        }
        if self.kind == Kind::Login
            && (self.value(Field::Identifier).trim().is_empty()
                || self.value(Field::Password).is_empty())
        {
            return Err("Identifier and password are required.".into());
        }
        Ok(())
    }
    fn secret(&self) -> Secret {
        let mut fields = BTreeMap::new();
        let mut insert = |field: Field, key: &str, trim: bool| {
            let value = self.value(field);
            fields.insert(
                key.to_owned(),
                if trim { value.trim() } else { value }.to_owned(),
            );
        };
        match self.kind {
            Kind::Login => {
                insert(Field::Identifier, "identifier", true);
                insert(Field::Password, "password", false);
                if !self.value(Field::Otp).trim().is_empty() {
                    insert(Field::Otp, "otp_secret", true);
                }
                fields.insert("identifier_type".into(), self.identifier_type.key().into());
            }
            Kind::Payment => {
                insert(Field::CardName, "cardholder_name", true);
                insert(Field::ExpMonth, "exp_month", true);
                insert(Field::ExpYear, "exp_year", true);
                insert(Field::Cvc, "cvc", false);
                insert(Field::Postal, "billing_postal_code", true);
                fields.insert(
                    "card_number".into(),
                    self.value(Field::CardNumber)
                        .chars()
                        .filter(|c| !c.is_whitespace())
                        .collect(),
                );
            }
            Kind::Address => {
                insert(Field::Line1, "address_line1", true);
                insert(Field::City, "city", true);
                insert(Field::Postal, "postal_code", true);
                insert(Field::Country, "country", true);
                if !self.value(Field::Line2).trim().is_empty() {
                    insert(Field::Line2, "address_line2", true);
                }
                if !self.value(Field::State).trim().is_empty() {
                    insert(Field::State, "state", true);
                }
            }
        }
        Secret {
            version: 1,
            kind: self.kind,
            fields,
        }
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Secret {
    version: u8,
    kind: Kind,
    fields: BTreeMap<String, String>,
}
impl Drop for Secret {
    fn drop(&mut self) {
        for value in self.fields.values_mut() {
            value.zeroize();
        }
    }
}

#[derive(Clone)]
pub enum Message {
    Refresh,
    Add,
    Edit(String),
    Kind(Kind),
    IdentifierType(IdentifierType),
    Field(Field, String),
    Save,
    Cancel,
    Delete(String),
    CancelDelete,
    ConfirmDelete,
    Loaded(u64, Result<Index, String>),
    Opened(u64, Result<Form, String>),
    Saved(u64, Result<Index, (String, Form)>),
    Removed(u64, Result<Index, String>),
}
impl fmt::Debug for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Iced diagnostics must never format field changes or decrypted forms.
        f.write_str(match self {
            Self::Refresh => "Refresh",
            Self::Add => "Add",
            Self::Edit(_) => "Edit",
            Self::Kind(_) => "Kind",
            Self::IdentifierType(_) => "IdentifierType",
            Self::Field(..) => "Field([redacted])",
            Self::Save => "Save",
            Self::Cancel => "Cancel",
            Self::Delete(_) => "Delete",
            Self::CancelDelete => "CancelDelete",
            Self::ConfirmDelete => "ConfirmDelete",
            Self::Loaded(..) => "Loaded",
            Self::Opened(..) => "Opened",
            Self::Saved(..) => "Saved",
            Self::Removed(..) => "Removed",
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Pending {
    Load,
    Open,
    Save,
    Delete,
}
#[derive(Default)]
pub struct State {
    root: Option<PathBuf>,
    index: Option<Index>,
    form: Option<Form>,
    editing: Option<Item>,
    pending_delete: Option<Item>,
    editor_open: bool,
    pending: Option<Pending>,
    generation: u64,
    error: Option<String>,
    feedback: Option<&'static str>,
}
impl State {
    pub fn has_dialog(&self) -> bool {
        self.editor_open || self.pending_delete.is_some()
    }
    pub fn dismiss(&mut self) {
        self.form = None;
        self.editing = None;
        self.pending_delete = None;
        self.editor_open = false;
        if self.pending == Some(Pending::Open) {
            self.generation = self.generation.wrapping_add(1);
            self.pending = None;
        }
    }
    fn begin(&mut self, kind: Pending) -> u64 {
        self.generation = self.generation.wrapping_add(1);
        self.pending = Some(kind);
        self.error = None;
        self.feedback = None;
        self.generation
    }
    fn item(&self, id: &str) -> Option<Item> {
        self.index
            .as_ref()?
            .items
            .iter()
            .find(|item| item.id == id)
            .cloned()
    }
    fn ready(&self) -> bool {
        cfg!(any(
            target_os = "macos",
            target_os = "windows",
            target_os = "linux"
        )) && self.index.is_some()
            && self.pending.is_none()
    }
}

pub fn update(app: &mut Hermes, message: Message) -> Task<AppMessage> {
    let root = app
        .library
        .as_ref()
        .map(|library| library.root().to_path_buf());
    let state = &mut app.vault;
    if root != state.root {
        *state = State {
            root: root.clone(),
            generation: state.generation.wrapping_add(1),
            ..State::default()
        };
    }
    let Some(root) = root else {
        state.error = Some("Open a local library to use Credential Vault.".into());
        return Task::none();
    };
    let task = match message {
        Message::Refresh if state.pending.is_none() => {
            state.dismiss();
            let request = state.begin(Pending::Load);
            Task::perform(blocking(move || load_index(&root)), move |result| {
                Message::Loaded(request, result)
            })
        }
        Message::Add if state.ready() => {
            state.dismiss();
            state.error = None;
            state.feedback = None;
            state.form = Some(Form::default());
            state.editor_open = true;
            Task::none()
        }
        Message::Edit(id) if state.ready() => {
            let Some(item) = state.item(&id) else {
                return Task::none();
            };
            let namespace = state.index.as_ref().unwrap().namespace.clone();
            state.dismiss();
            state.editor_open = true;
            state.editing = Some(item.clone());
            let request = state.begin(Pending::Open);
            Task::perform(
                blocking(move || open_item(&namespace, &item)),
                move |result| Message::Opened(request, result),
            )
        }
        Message::Kind(kind) if state.pending.is_none() && state.editing.is_none() => {
            if let Some(form) = &mut state.form {
                form.kind = kind;
            }
            Task::none()
        }
        Message::IdentifierType(kind) if state.pending.is_none() => {
            if let Some(form) = &mut state.form {
                form.identifier_type = kind;
            }
            Task::none()
        }
        Message::Field(field, value) if state.pending.is_none() => {
            if let Some(form) = &mut state.form {
                form.set(field, value);
            }
            state.error = None;
            Task::none()
        }
        Message::Save if state.ready() => {
            let Some(form) = state.form.as_ref() else {
                return Task::none();
            };
            if let Err(error) = form.validate() {
                state.error = Some(error);
                return Task::none();
            }
            let form = state.form.take().unwrap();
            let editing = state.editing.clone();
            let namespace = state.index.as_ref().unwrap().namespace.clone();
            let request = state.begin(Pending::Save);
            Task::perform(
                async move {
                    match tokio::task::spawn_blocking(move || {
                    match save_item(&root, &namespace, editing.as_ref(), &form) {
                        Ok(index) => Ok(index),
                        Err(error) => Err((error, form)),
                    }
                }).await {
                    Ok(result) => result,
                    Err(_) => Err(("Credential Vault stopped before the save could be confirmed. Refresh before retrying.".into(), Form::default())),
                }
                },
                move |result| Message::Saved(request, result),
            )
        }
        Message::Cancel | Message::CancelDelete => {
            state.dismiss();
            Task::none()
        }
        Message::Delete(id) if state.ready() => {
            state.dismiss();
            state.pending_delete = state.item(&id);
            state.error = None;
            Task::none()
        }
        Message::ConfirmDelete if state.ready() => {
            let Some(item) = state.pending_delete.take() else {
                return Task::none();
            };
            let namespace = state.index.as_ref().unwrap().namespace.clone();
            let request = state.begin(Pending::Delete);
            Task::perform(
                blocking(move || remove_item(&root, &namespace, &item)),
                move |result| Message::Removed(request, result),
            )
        }
        Message::Loaded(request, result) if request == state.generation => {
            state.pending = None;
            match result {
                Ok(index) => {
                    state.index = Some(index);
                    state.error = None;
                }
                Err(error) => {
                    state.index = None;
                    state.error = Some(error);
                }
            }
            Task::none()
        }
        Message::Opened(request, result) if request == state.generation => {
            state.pending = None;
            if state.editor_open {
                match result {
                    Ok(form) => state.form = Some(form),
                    Err(error) => {
                        state.dismiss();
                        state.error = Some(error);
                    }
                }
            }
            Task::none()
        }
        Message::Saved(request, result) if request == state.generation => {
            state.pending = None;
            match result {
                Ok(index) => {
                    state.index = Some(index);
                    state.dismiss();
                    state.feedback = Some("Saved.");
                }
                Err((error, form)) => {
                    state.error = Some(error);
                    if state.editor_open {
                        state.form = Some(form);
                    }
                }
            }
            Task::none()
        }
        Message::Removed(request, result) if request == state.generation => {
            state.pending = None;
            match result {
                Ok(index) => {
                    state.index = Some(index);
                    state.feedback = Some("Removed saved item.");
                }
                Err(error) => state.error = Some(error),
            }
            Task::none()
        }
        _ => Task::none(),
    };
    task.map(AppMessage::Vault)
}

async fn blocking<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tokio::task::spawn_blocking(operation)
        .await
        .unwrap_or_else(|_| Err("Credential Vault operation could not finish. Try again.".into()))
}
fn new_id() -> String {
    Uuid::new_v4().simple().to_string()
}
fn valid_id(value: &str) -> bool {
    value.len() == 32 && value.bytes().all(|b| b.is_ascii_hexdigit())
}
fn account(namespace: &str, id: &str) -> Result<String, String> {
    if !valid_id(namespace) || !valid_id(id) {
        return Err("Invalid vault item identity.".into());
    }
    Ok(format!("{namespace}:{id}"))
}
fn read_index(root: &Path) -> Result<Option<Index>, String> {
    let mut file = match File::open(root.join(INDEX_FILE)) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => {
            return Err(
                "Could not read the local vault index. Check this library's file permissions."
                    .into(),
            );
        }
    };
    let mut bytes = Vec::new();
    (&mut file)
        .take(MAX_INDEX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "Could not read the local vault index.")?;
    if bytes.len() as u64 > MAX_INDEX_BYTES {
        return Err("The local vault index is too large to open.".into());
    }
    let index: Index = serde_json::from_slice(&bytes)
        .map_err(|_| "The local vault index is unreadable. It has not been replaced.")?;
    index.validate()?;
    Ok(Some(index))
}
fn lock_index(root: &Path) -> Result<File, String> {
    let mut options = OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let file = options
        .open(root.join(LOCK_FILE))
        .map_err(|_| "Could not open the local vault lock.")?;
    file.lock_exclusive()
        .map_err(|_| "Could not lock the local vault index.")?;
    Ok(file)
}
fn load_index(root: &Path) -> Result<Index, String> {
    let _lock = lock_index(root)?;
    Ok(read_index(root)?.unwrap_or_else(Index::empty))
}
fn write_index(root: &Path, index: &Index) -> Result<(), String> {
    index.validate()?;
    let bytes =
        serde_json::to_vec_pretty(index).map_err(|_| "Could not encode the local vault index.")?;
    if bytes.len() as u64 > MAX_INDEX_BYTES {
        return Err("The local vault index has reached its size limit.".into());
    }
    let mut temp =
        NamedTempFile::new_in(root).map_err(|_| "Could not create the local vault index.")?;
    temp.write_all(&bytes)
        .and_then(|_| temp.as_file().sync_all())
        .map_err(|_| "Could not write the local vault index.")?;
    temp.persist(root.join(INDEX_FILE))
        .map_err(|_| "Could not commit the local vault index.")?;
    #[cfg(unix)]
    File::open(root)
        .and_then(|dir| dir.sync_all())
        .map_err(|_| "The vault index was written, but its disk sync could not be confirmed.")?;
    Ok(())
}
fn current_index(root: &Path, namespace: &str) -> Result<Index, String> {
    let index = read_index(root)?.unwrap_or_else(|| Index {
        version: 1,
        namespace: namespace.to_owned(),
        items: Vec::new(),
    });
    if index.namespace != namespace {
        return Err("The vault owner changed. Refresh before trying again.".into());
    }
    Ok(index)
}
fn unchanged(index: &Index, expected: &Item) -> Result<usize, String> {
    index
        .items
        .iter()
        .position(|item| item.id == expected.id && item.updated_at == expected.updated_at)
        .ok_or_else(|| "This item changed in another window. Refresh and open it again.".into())
}
fn open_item(namespace: &str, item: &Item) -> Result<Form, String> {
    let bytes = keychain::read(&account(namespace, &item.id)?)?;
    let mut secret: Secret = serde_json::from_slice(&bytes)
        .map_err(|_| "This saved item has an unreadable credential payload.")?;
    if secret.version != 1 || secret.kind != item.kind {
        return Err("This saved item has an unsupported credential format.".into());
    }
    let mut form = Form::default();
    form.kind = item.kind;
    form.set(Field::Label, item.label.clone());
    form.set(Field::Origin, item.origin.clone());
    form.identifier_type = match secret.fields.get("identifier_type").map(String::as_str) {
        Some("phone") => IdentifierType::Phone,
        Some("username") => IdentifierType::Username,
        _ => IdentifierType::Email,
    };
    for field in [
        Field::Identifier,
        Field::Password,
        Field::Otp,
        Field::CardNumber,
        Field::CardName,
        Field::ExpMonth,
        Field::ExpYear,
        Field::Cvc,
        Field::Postal,
        Field::Line1,
        Field::Line2,
        Field::City,
        Field::State,
        Field::Country,
    ] {
        let key = if field == Field::Postal && item.kind == Kind::Payment {
            "billing_postal_code"
        } else {
            field.key()
        };
        if let Some(value) = secret.fields.remove(key) {
            form.set(field, value);
        }
    }
    Ok(form)
}
fn save_item(
    root: &Path,
    namespace: &str,
    editing: Option<&Item>,
    form: &Form,
) -> Result<Index, String> {
    form.validate()?;
    let _lock = lock_index(root)?;
    let before = current_index(root, namespace)?;
    let mut after = before.clone();
    let position = editing.map(|item| unchanged(&before, item)).transpose()?;
    if editing.is_some_and(|item| item.kind != form.kind) {
        return Err("Create a new item to use a different kind.".into());
    }
    let now = Utc::now().to_rfc3339();
    let item = Item {
        id: editing.map(|item| item.id.clone()).unwrap_or_else(new_id),
        kind: form.kind,
        label: form.value(Field::Label).trim().to_owned(),
        origin: form.value(Field::Origin).trim().to_owned(),
        created_at: editing
            .map(|item| item.created_at.clone())
            .unwrap_or_else(|| now.clone()),
        updated_at: now,
        identifier: (form.kind == Kind::Login)
            .then(|| form.value(Field::Identifier).trim().to_owned()),
        identifier_type: (form.kind == Kind::Login).then(|| form.identifier_type.key().to_owned()),
        has_otp: form.kind == Kind::Login && !form.value(Field::Otp).trim().is_empty(),
    };
    let name = account(namespace, &item.id)?;
    let secret = Zeroizing::new(
        serde_json::to_vec(&form.secret()).map_err(|_| "Could not encode this credential.")?,
    );
    if secret.len() > MAX_SECRET_BYTES {
        return Err("This credential is too large to save.".into());
    }
    if let Some(position) = position {
        after.items[position] = item;
    } else {
        after.items.push(item);
    }
    // Persist only metadata before touching Keychain so a crash never strands an
    // undiscoverable secret. The lock hides the provisional index from this app.
    write_index(root, &after)?;
    if let Err(error) = keychain::write(&name, &secret, editing.is_some()) {
        if write_index(root, &before).is_err() {
            return Err(format!(
                "{error} The local index could not be restored; refresh before retrying."
            ));
        }
        return Err(error);
    }
    Ok(after)
}
fn remove_item(root: &Path, namespace: &str, expected: &Item) -> Result<Index, String> {
    let _lock = lock_index(root)?;
    let mut index = current_index(root, namespace)?;
    let position = unchanged(&index, expected)?;
    keychain::remove(&account(namespace, &expected.id)?)?;
    index.items.remove(position);
    write_index(root, &index).map_err(|_| "The credential was deleted, but its local listing could not be removed. Refresh and delete the listing again.")?;
    Ok(index)
}

pub fn view(app: &Hermes) -> Element<'_, AppMessage> {
    let state = &app.vault;
    let p = ui::palette(app.dark);
    let busy = state.pending.is_some();
    let count = state.index.as_ref().map_or(0, |index| index.items.len());
    let mut add = button(text("Add").size(12))
        .padding([5, 12])
        .style(button::secondary);
    if state.ready() && !state.has_dialog() {
        add = add.on_press(AppMessage::Vault(Message::Add));
    }
    let mut refresh = button(text("Refresh").size(11)).style(button::text);
    if !busy {
        refresh = refresh.on_press(AppMessage::Vault(Message::Refresh));
    }
    let mut body = column![
        row![column![text("Passwords & Logins").size(18), text(format!("{count} saved")).size(11).color(p.muted)].spacing(4), Space::new().width(Fill), refresh, add].spacing(8).align_y(Alignment::Center),
        text(format!("Store a login, payment card or address in {STORE_NAME}. Secrets stay on this device and outside your conversations.")).size(12).color(p.muted),
    ].spacing(14);
    if !cfg!(any(
        target_os = "macos",
        target_os = "windows",
        target_os = "linux"
    )) {
        body = body.push(
            text("Credential Vault is available on macOS, Windows and Linux.")
                .size(12)
                .color(p.muted),
        );
    }
    if cfg!(target_os = "linux") {
        body = body.push(text("Linux needs an unlocked Secret Service keyring, such as GNOME Keyring or KWallet, with a default collection.").size(11).color(p.muted));
    }
    if let Some(error) = &state.error {
        body = body.push(
            text(error)
                .size(12)
                .color(iced::Color::from_rgb8(190, 65, 65)),
        );
    }
    if let Some(feedback) = state.feedback {
        body = body.push(text(feedback).size(12).color(p.muted));
    }
    if state.editor_open {
        body = body.push(editor(app));
    } else if let Some(item) = &state.pending_delete {
        body = body.push(
            container(
                column![
                    text("Delete this item?").size(15),
                    text(format!(
                        "“{}” will be removed. This cannot be undone.",
                        item.label
                    ))
                    .size(12),
                    row![
                        Space::new().width(Fill),
                        button(text("Cancel").size(12))
                            .on_press(AppMessage::Vault(Message::CancelDelete))
                            .style(button::text),
                        button(text("Delete").size(12))
                            .on_press(AppMessage::Vault(Message::ConfirmDelete))
                            .style(button::danger)
                    ]
                    .spacing(8),
                ]
                .spacing(12),
            )
            .padding(16)
            .width(Fill)
            .style(move |_| container::Style {
                background: Some(p.card.into()),
                border: Border {
                    radius: 8.0.into(),
                    width: 1.0,
                    color: p.border,
                },
                ..Default::default()
            }),
        );
    }
    if state.pending == Some(Pending::Load) {
        body = body.push(text("Loading saved items…").size(12).color(p.muted));
    } else if state.pending == Some(Pending::Delete) {
        body = body.push(text("Removing saved item…").size(12).color(p.muted));
    } else if state.index.is_none() && !busy {
        body = body.push(
            button(text("Load saved items").size(12))
                .on_press(AppMessage::Vault(Message::Refresh))
                .style(button::secondary),
        );
    } else if count == 0 && !state.editor_open {
        body = body.push(
            container(
                column![
                    text("Nothing saved yet").size(14),
                    text("Use Add to save a login, card or address.")
                        .size(12)
                        .color(p.muted)
                ]
                .spacing(6),
            )
            .padding([22, 0]),
        );
    }
    if let Some(index) = &state.index {
        for item in &index.items {
            let mut detail = format!("{} · {}", item.kind, item.origin);
            if let Some(identifier) = &item.identifier {
                detail = format!("{identifier} · {detail}");
            }
            let date = item.created_at.get(..10).unwrap_or(&item.created_at);
            let mut title = row![text(&item.label).size(13)]
                .spacing(8)
                .align_y(Alignment::Center);
            if item.has_otp {
                title = title.push(text("Authenticator key saved").size(10).color(p.muted));
            }
            let mut edit = button(text("Edit").size(11)).style(button::text);
            let mut delete = button(text("Remove").size(11)).style(button::text);
            if state.ready() && !state.has_dialog() {
                edit = edit.on_press(AppMessage::Vault(Message::Edit(item.id.clone())));
                delete = delete.on_press(AppMessage::Vault(Message::Delete(item.id.clone())));
            }
            body = body.push(
                container(
                    row![
                        column![
                            title,
                            text(detail).size(11).color(p.muted),
                            text(format!("Added {date}")).size(10).color(p.muted)
                        ]
                        .spacing(4)
                        .width(Fill),
                        edit,
                        delete,
                    ]
                    .spacing(8)
                    .align_y(Alignment::Center),
                )
                .padding([12, 0]),
            );
            body = body.push(rule::horizontal(1));
        }
    }
    body.push(text("Native development vault · Browser autofill, agent access and password manager connections are not connected yet.").size(11).color(p.muted)).width(Fill).into()
}
fn editor(app: &Hermes) -> Element<'_, AppMessage> {
    let state = &app.vault;
    let p = ui::palette(app.dark);
    let title = if state.editing.is_some() {
        "Edit saved item"
    } else {
        "Add a login, card or address"
    };
    let mut body = column![
        text(title).size(15),
        text(format!(
            "Stored encrypted in this app’s {STORE_NAME} entries."
        ))
        .size(11)
        .color(p.muted)
    ]
    .spacing(12);
    if let Some(form) = &state.form {
        let kind: Element<'_, AppMessage> = if state.editing.is_none() {
            pick_list(
                [Kind::Login, Kind::Payment, Kind::Address],
                Some(form.kind),
                |kind| AppMessage::Vault(Message::Kind(kind)),
            )
            .text_size(12)
            .into()
        } else {
            text(form.kind.to_string()).size(12).into()
        };
        body = body.push(column![text("Kind").size(11).color(p.muted), kind].spacing(5));
        body = body.push(input(
            form,
            Field::Label,
            "Label",
            "e.g. GitHub work account",
            false,
        ));
        body = body.push(input(
            form,
            Field::Origin,
            "Site origin",
            if form.kind == Kind::Login {
                "https://github.com"
            } else {
                "https://shop.example.com"
            },
            false,
        ));
        match form.kind {
            Kind::Login => {
                body = body.push(
                    row![
                        column![
                            text("Identifier type").size(11).color(p.muted),
                            pick_list(
                                [
                                    IdentifierType::Email,
                                    IdentifierType::Phone,
                                    IdentifierType::Username
                                ],
                                Some(form.identifier_type),
                                |kind| AppMessage::Vault(Message::IdentifierType(kind))
                            )
                            .text_size(12)
                        ]
                        .spacing(5),
                        input(form, Field::Identifier, "Identifier", "", false),
                    ]
                    .spacing(12)
                    .align_y(Alignment::End),
                );
                body = body.push(input(form, Field::Password, "Password", "", true));
                body = body.push(input(
                    form,
                    Field::Otp,
                    "Authenticator key (optional)",
                    "Base32 secret or otpauth:// link",
                    true,
                ));
                body = body.push(text("The setup key supplied when you enable 2FA. This native build stores the key; code generation is not connected yet.").size(10).color(p.muted));
            }
            Kind::Payment => {
                body = body.push(input(form, Field::CardNumber, "Card number", "", true));
                body = body.push(input(form, Field::CardName, "Name on card", "", false));
                body = body.push(
                    row![
                        input(form, Field::ExpMonth, "Exp. month", "MM", false),
                        input(form, Field::ExpYear, "Exp. year", "YYYY", false)
                    ]
                    .spacing(12),
                );
                body = body.push(
                    row![
                        input(form, Field::Cvc, "CVC", "", true),
                        input(form, Field::Postal, "Postal code", "", false)
                    ]
                    .spacing(12),
                );
            }
            Kind::Address => {
                body = body.push(input(form, Field::Line1, "Address line 1", "", false));
                body = body.push(input(
                    form,
                    Field::Line2,
                    "Address line 2 (optional)",
                    "",
                    false,
                ));
                body = body.push(
                    row![
                        input(form, Field::City, "City", "", false),
                        input(form, Field::State, "State / region (optional)", "", false)
                    ]
                    .spacing(12),
                );
                body = body.push(
                    row![
                        input(form, Field::Postal, "Postal code", "", false),
                        input(form, Field::Country, "Country", "", false)
                    ]
                    .spacing(12),
                );
            }
        }
    } else {
        body = body.push(
            text(if state.pending == Some(Pending::Open) {
                "Opening selected credential…"
            } else {
                "Saving…"
            })
            .size(12)
            .color(p.muted),
        );
    }
    let mut actions = row![
        Space::new().width(Fill),
        button(
            text(if state.pending == Some(Pending::Save) {
                "Close"
            } else {
                "Cancel"
            })
            .size(12)
        )
        .on_press(AppMessage::Vault(Message::Cancel))
        .style(button::text)
    ]
    .spacing(8);
    if state.form.is_some() && state.pending.is_none() {
        actions = actions.push(
            button(text("Save").size(12))
                .on_press(AppMessage::Vault(Message::Save))
                .style(button::primary),
        );
    }
    container(body.push(actions))
        .padding(16)
        .width(Fill)
        .style(move |_| container::Style {
            background: Some(p.card.into()),
            border: Border {
                radius: 8.0.into(),
                width: 1.0,
                color: p.border,
            },
            ..Default::default()
        })
        .into()
}
fn input<'a>(
    form: &'a Form,
    field: Field,
    label: &'static str,
    placeholder: &'static str,
    secure: bool,
) -> Element<'a, AppMessage> {
    column![
        text(label).size(11),
        text_input(placeholder, form.value(field))
            .secure(secure)
            .on_input(move |value| AppMessage::Vault(Message::Field(field, value)))
            .size(13)
            .padding(8),
    ]
    .spacing(5)
    .width(Fill)
    .into()
}

#[cfg(target_os = "macos")]
mod keychain {
    use super::{KEYCHAIN_SERVICE, MAX_SECRET_BYTES};
    use core_foundation::{
        base::{CFType, CFTypeRef, TCFType},
        boolean::CFBoolean,
        data::CFData,
        dictionary::{CFDictionary, CFDictionaryRef},
        string::{CFString, CFStringRef},
    };
    use std::ptr;
    use zeroize::Zeroizing;

    #[link(name = "Security", kind = "framework")]
    unsafe extern "C" {
        static kSecClass: CFStringRef;
        static kSecClassGenericPassword: CFStringRef;
        static kSecAttrService: CFStringRef;
        static kSecAttrAccount: CFStringRef;
        static kSecAttrSynchronizable: CFStringRef;
        static kSecValueData: CFStringRef;
        static kSecReturnData: CFStringRef;
        static kSecMatchLimit: CFStringRef;
        static kSecMatchLimitOne: CFStringRef;
        fn SecItemAdd(attributes: CFDictionaryRef, result: *mut CFTypeRef) -> i32;
        fn SecItemCopyMatching(query: CFDictionaryRef, result: *mut CFTypeRef) -> i32;
        fn SecItemUpdate(query: CFDictionaryRef, attributes: CFDictionaryRef) -> i32;
        fn SecItemDelete(query: CFDictionaryRef) -> i32;
    }
    const NOT_FOUND: i32 = -25300;
    fn string(reference: CFStringRef) -> CFString {
        // These framework-owned constants remain alive for the process lifetime.
        unsafe { CFString::wrap_under_get_rule(reference) }
    }
    fn query(account: &str) -> Vec<(CFString, CFType)> {
        unsafe {
            vec![
                (
                    string(kSecClass),
                    string(kSecClassGenericPassword).as_CFType(),
                ),
                (
                    string(kSecAttrService),
                    CFString::new(KEYCHAIN_SERVICE).as_CFType(),
                ),
                (string(kSecAttrAccount), CFString::new(account).as_CFType()),
                (
                    string(kSecAttrSynchronizable),
                    CFBoolean::false_value().as_CFType(),
                ),
            ]
        }
    }
    fn status(code: i32) -> Result<(), String> {
        if code == 0 {
            return Ok(());
        }
        Err(match code {
            -128 | -25293 => "macOS Keychain access was cancelled or denied. You can try again.".into(),
            -25308 => "macOS Keychain is locked or unavailable. Unlock it, then try again.".into(),
            NOT_FOUND => "The named credential is missing from macOS Keychain. Remove its listing and add it again.".into(),
            -25299 => "This named credential already exists. Refresh before trying again.".into(),
            _ => format!("macOS Keychain could not complete this operation (status {code})."),
        })
    }
    pub fn read(account: &str) -> Result<Zeroizing<Vec<u8>>, String> {
        let mut pairs = query(account);
        unsafe {
            pairs.push((string(kSecReturnData), CFBoolean::true_value().as_CFType()));
            pairs.push((
                string(kSecMatchLimit),
                string(kSecMatchLimitOne).as_CFType(),
            ));
        }
        let query = CFDictionary::from_CFType_pairs(&pairs);
        let mut result: CFTypeRef = ptr::null();
        let code = unsafe { SecItemCopyMatching(query.as_concrete_TypeRef(), &mut result) };
        // Even unexpected error results must release the framework-owned object.
        let result = (!result.is_null()).then(|| unsafe { CFType::wrap_under_create_rule(result) });
        status(code)?;
        let data = result
            .and_then(|value| value.downcast_into::<CFData>())
            .ok_or("macOS Keychain returned an unsupported item format.")?;
        if data.len() < 0 || data.len() as usize > MAX_SECRET_BYTES {
            return Err("This Keychain credential is too large to open.".into());
        }
        Ok(Zeroizing::new(data.bytes().to_vec()))
    }
    pub fn write(account: &str, secret: &[u8], edit: bool) -> Result<(), String> {
        let mut pairs = query(account);
        let value = CFData::from_buffer(secret);
        let data = unsafe { (string(kSecValueData), value.as_CFType()) };
        let code = if edit {
            let query = CFDictionary::from_CFType_pairs(&pairs);
            let update = CFDictionary::from_CFType_pairs(&[data]);
            unsafe { SecItemUpdate(query.as_concrete_TypeRef(), update.as_concrete_TypeRef()) }
        } else {
            pairs.push(data);
            let attributes = CFDictionary::from_CFType_pairs(&pairs);
            unsafe { SecItemAdd(attributes.as_concrete_TypeRef(), ptr::null_mut()) }
        };
        status(code)
    }
    pub fn remove(account: &str) -> Result<(), String> {
        let query = CFDictionary::from_CFType_pairs(&query(account));
        let code = unsafe { SecItemDelete(query.as_concrete_TypeRef()) };
        // A prior successful delete whose metadata commit failed is retryable.
        if code == NOT_FOUND {
            Ok(())
        } else {
            status(code)
        }
    }
}
#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
mod keychain {
    use zeroize::Zeroizing;
    pub fn read(_: &str) -> Result<Zeroizing<Vec<u8>>, String> {
        Err("No native credential store is supported on this operating system.".into())
    }
    pub fn write(_: &str, _: &[u8], _: bool) -> Result<(), String> {
        Err("No native credential store is supported on this operating system.".into())
    }
    pub fn remove(_: &str) -> Result<(), String> {
        Err("No native credential store is supported on this operating system.".into())
    }
}

#[cfg(target_os = "windows")]
mod keychain {
    use super::{KEYCHAIN_SERVICE, MAX_SECRET_BYTES};
    use std::{ptr, slice};
    use windows_sys::Win32::{
        Foundation::{
            ERROR_ACCESS_DENIED, ERROR_CANCELLED, ERROR_NO_SUCH_LOGON_SESSION, ERROR_NOT_FOUND,
            GetLastError,
        },
        Security::Credentials::{
            CRED_MAX_CREDENTIAL_BLOB_SIZE, CRED_PERSIST_LOCAL_MACHINE, CRED_TYPE_GENERIC,
            CREDENTIALW, CredDeleteW, CredFree, CredReadW, CredWriteW,
        },
    };
    use zeroize::Zeroizing;

    // Each target names this app's service and one library/item account. Never
    // enumerate Credential Manager or select another application's credentials.
    fn target(account: &str) -> Vec<u16> {
        format!("{KEYCHAIN_SERVICE}/{account}")
            .encode_utf16()
            .chain(Some(0))
            .collect()
    }
    fn error(code: u32) -> String {
        match code {
            ERROR_ACCESS_DENIED | ERROR_CANCELLED => "Windows Credential Manager access was denied or cancelled. You can try again.".into(),
            ERROR_NO_SUCH_LOGON_SESSION => "Windows Credential Manager is unavailable for this logon session. Sign in normally, then try again.".into(),
            _ => format!("Windows Credential Manager could not complete this operation (error {code})."),
        }
    }
    struct Credential(*mut CREDENTIALW);
    impl Drop for Credential {
        fn drop(&mut self) {
            // CredReadW returns a writable allocation owned by CredFree.
            unsafe {
                if !self.0.is_null() {
                    let value = &mut *self.0;
                    if !value.CredentialBlob.is_null()
                        && value.CredentialBlobSize <= CRED_MAX_CREDENTIAL_BLOB_SIZE
                    {
                        ptr::write_bytes(
                            value.CredentialBlob,
                            0,
                            value.CredentialBlobSize as usize,
                        );
                    }
                    CredFree(self.0.cast());
                }
            }
        }
    }
    fn lookup(target: &[u16]) -> Result<Option<Credential>, String> {
        let mut credential = ptr::null_mut();
        let success = unsafe { CredReadW(target.as_ptr(), CRED_TYPE_GENERIC, 0, &mut credential) };
        if success == 0 {
            let code = unsafe { GetLastError() };
            if code == ERROR_NOT_FOUND {
                return Ok(None);
            }
            return Err(error(code));
        }
        if credential.is_null() {
            return Err("Windows Credential Manager returned an empty credential.".into());
        }
        Ok(Some(Credential(credential)))
    }
    pub fn read(account: &str) -> Result<Zeroizing<Vec<u8>>, String> {
        let target = target(account);
        let credential = lookup(&target)?.ok_or("The named credential is missing from Windows Credential Manager. Remove its listing and add it again.")?;
        let value = unsafe { &*credential.0 };
        if value.CredentialBlobSize as usize > MAX_SECRET_BYTES {
            return Err("This saved credential exceeds the Windows size limit.".into());
        }
        if value.CredentialBlobSize == 0 {
            return Ok(Zeroizing::new(Vec::new()));
        }
        if value.CredentialBlob.is_null() {
            return Err("Windows Credential Manager returned an unreadable credential.".into());
        }
        // The allocation remains owned by `credential` until after the copy.
        Ok(Zeroizing::new(
            unsafe {
                slice::from_raw_parts(value.CredentialBlob, value.CredentialBlobSize as usize)
            }
            .to_vec(),
        ))
    }
    pub fn write(account: &str, secret: &[u8], edit: bool) -> Result<(), String> {
        if secret.len() > CRED_MAX_CREDENTIAL_BLOB_SIZE as usize {
            return Err(
                "This credential exceeds Windows Credential Manager's 2,560-byte limit.".into(),
            );
        }
        let mut target = target(account);
        let exists = lookup(&target)?.is_some();
        if edit && !exists {
            return Err("The named credential is missing from Windows Credential Manager. Remove its listing and add it again.".into());
        }
        if !edit && exists {
            return Err(
                "This named credential already exists. Refresh before trying again.".into(),
            );
        }
        let mut username: Vec<u16> = "Hermes native vault"
            .encode_utf16()
            .chain(Some(0))
            .collect();
        let credential = CREDENTIALW {
            Type: CRED_TYPE_GENERIC,
            TargetName: target.as_mut_ptr(),
            CredentialBlobSize: secret.len() as u32,
            CredentialBlob: secret.as_ptr().cast_mut(),
            Persist: CRED_PERSIST_LOCAL_MACHINE,
            UserName: username.as_mut_ptr(),
            ..CREDENTIALW::default()
        };
        if unsafe { CredWriteW(&credential, 0) } == 0 {
            Err(error(unsafe { GetLastError() }))
        } else {
            Ok(())
        }
    }
    pub fn remove(account: &str) -> Result<(), String> {
        let target = target(account);
        if unsafe { CredDeleteW(target.as_ptr(), CRED_TYPE_GENERIC, 0) } == 0 {
            let code = unsafe { GetLastError() };
            if code == ERROR_NOT_FOUND {
                Ok(())
            } else {
                Err(error(code))
            }
        } else {
            Ok(())
        }
    }
}

#[cfg(target_os = "linux")]
mod keychain {
    use super::{KEYCHAIN_SERVICE, MAX_SECRET_BYTES};
    use secret_service::{
        EncryptionType,
        blocking::{Item, SecretService},
    };
    use std::collections::HashMap;
    use zeroize::Zeroizing;

    fn attributes(account: &str) -> HashMap<&str, &str> {
        HashMap::from([("application", KEYCHAIN_SERVICE), ("account", account)])
    }
    fn connect<'a>() -> Result<SecretService<'a>, String> {
        // Refuse a plaintext D-Bus secret session. No session-bus connection is
        // opened at startup or for listing the metadata in the native UI.
        SecretService::connect(EncryptionType::Dh).map_err(|_| "Could not connect securely to Linux Secret Service. Start a keyring such as GNOME Keyring or KWallet, then try again.".into())
    }
    fn named<'a>(
        service: &'a SecretService<'a>,
        account: &str,
    ) -> Result<Option<Item<'a>>, String> {
        let found = service
            .search_items(attributes(account))
            .map_err(|_| "Linux Secret Service could not find this named credential.")?;
        let mut matches = found.unlocked.into_iter().chain(found.locked);
        let item = matches.next();
        if matches.next().is_some() {
            return Err("The keyring has multiple copies of this app's named credential. Resolve the duplicate in your keyring before retrying.".into());
        }
        Ok(item)
    }
    fn unlock(item: &Item<'_>) -> Result<(), String> {
        if item
            .is_locked()
            .map_err(|_| "Could not read the selected keyring item's lock state.")?
        {
            item.unlock().map_err(|_| "The selected keyring item could not be unlocked. Approve your keyring's prompt, then try again.")?;
        }
        item.ensure_unlocked()
            .map_err(|_| "The selected keyring item is still locked.".into())
    }
    pub fn read(account: &str) -> Result<Zeroizing<Vec<u8>>, String> {
        let service = connect()?;
        let item = named(&service, account)?.ok_or("The named credential is missing from Linux Secret Service. Remove its listing and add it again.")?;
        unlock(&item)?;
        let secret = Zeroizing::new(
            item.get_secret()
                .map_err(|_| "Linux Secret Service could not read this credential.")?,
        );
        if secret.len() > MAX_SECRET_BYTES {
            return Err("This saved keyring credential is too large to open.".into());
        }
        Ok(secret)
    }
    pub fn write(account: &str, secret: &[u8], edit: bool) -> Result<(), String> {
        let service = connect()?;
        let existing = named(&service, account)?;
        if edit {
            let item = existing.ok_or("The named credential is missing from Linux Secret Service. Remove its listing and add it again.")?;
            unlock(&item)?;
            return item
                .set_secret(secret, "application/json")
                .map_err(|_| "Linux Secret Service could not update this credential.".into());
        }
        if existing.is_some() {
            return Err(
                "This named credential already exists. Refresh before trying again.".into(),
            );
        }
        let collection = service.get_default_collection().map_err(|_| "Linux Secret Service has no default collection. Create a default keyring in your password manager, then try again.")?;
        if collection
            .is_locked()
            .map_err(|_| "Could not read the default keyring's lock state.")?
        {
            collection.unlock().map_err(|_| "The default keyring could not be unlocked. Approve your keyring's prompt, then try again.")?;
        }
        collection
            .ensure_unlocked()
            .map_err(|_| "The default keyring is still locked.")?;
        collection
            .create_item(
                "Hermes native vault credential",
                attributes(account),
                secret,
                false,
                "application/json",
            )
            .map_err(|_| "Linux Secret Service could not save this credential.")?;
        Ok(())
    }
    pub fn remove(account: &str) -> Result<(), String> {
        let service = connect()?;
        let Some(item) = named(&service, account)? else {
            return Ok(());
        };
        unlock(&item)?;
        item.delete()
            .map_err(|_| "Linux Secret Service could not remove this credential.".into())
    }
}
