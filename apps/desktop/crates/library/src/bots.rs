//! Bot Mode (plugins/hermes-bots): one bot = one profile with a persistent
//! identity and ONE canonical forever-chat, identified by NAME. The chat's
//! identity is (bot, session titled exactly "Bot Chat"); there is no stored
//! session-id pointer and recency never picks it. Bot chats are hidden from
//! the Sessions list, so the bot row is the only door.
use super::*;

pub const BOT_CHAT_TITLE: &str = "Bot Chat";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Bot {
    /// Profile name: lowercase letters, digits, `-` and `_`.
    pub name: String,
    pub display_name: String,
    #[serde(default)]
    pub emoji: String,
    #[serde(default)]
    pub description: String,
    pub created_at: String,
}

#[derive(Default, Serialize, Deserialize)]
struct Roster { #[serde(default)] bots: Vec<Bot> }

/// "Research Helper" → "research-helper".
fn profile_name(display: &str) -> String {
    let mut out = String::new();
    for c in display.trim().chars().flat_map(char::to_lowercase) {
        if c.is_ascii_alphanumeric() { out.push(c); } else if !out.ends_with('-') && !out.is_empty() { out.push('-'); }
    }
    out.trim_end_matches('-').chars().take(48).collect()
}

/// The SOUL a bot is born with (soul.ts), written to its profile's SOUL.md.
fn soul(bot: &Bot, instructions: &str) -> String {
    let mut text = format!("# {}\n\nYou are {}, a named Hermes agent.", bot.display_name, bot.display_name);
    if !bot.description.trim().is_empty() { text.push_str(&format!(" {}", bot.description.trim())); }
    if !instructions.trim().is_empty() { text.push_str(&format!("\n\n## How you work\n\n{}", instructions.trim())); }
    text.push('\n');
    text
}

impl Library {
    fn roster_path(&self) -> PathBuf { self.root.join("bots.json") }
    fn read_roster(&self) -> Result<Roster> {
        match fs::read(self.roster_path()) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("Could not read the bot roster"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Roster::default()),
            Err(e) => Err(e.into()),
        }
    }
    /// The bot's own profile folder: SOUL.md, settings, memory and skills.
    pub fn bot_profile(&self, name: &str) -> Result<PathBuf> {
        ensure!(safe_id(name), "Invalid bot name");
        Ok(self.root.join("profiles").join(name))
    }
    pub fn bots(&self) -> Result<Vec<Bot>> { Ok(self.read_roster()?.bots) }

    pub fn create_bot(&self, display_name: &str, description: &str, instructions: &str, emoji: &str) -> Result<Bot> {
        let display_name = display_name.trim();
        ensure!(!display_name.is_empty(), "Give the bot a name");
        ensure!(display_name.chars().count() <= 60, "A bot name must contain at most 60 characters");
        let name = profile_name(display_name);
        ensure!(!name.is_empty() && safe_id(&name), "Use letters or digits in the bot's name");
        ensure!(name != "default", "\"default\" is reserved for your main profile");
        let mut roster = self.read_roster()?;
        ensure!(!roster.bots.iter().any(|bot| bot.name == name), "A bot named {display_name} already exists");
        let bot = Bot { name: name.clone(), display_name: display_name.into(), emoji: emoji.trim().chars().take(4).collect(), description: description.trim().into(), created_at: now() };
        let profile = self.bot_profile(&name)?;
        fs::create_dir_all(&profile)?;
        fs::write(profile.join("SOUL.md"), soul(&bot, instructions))?;
        // A bot starts from the main profile's settings, then owns its own copy.
        let main_settings = self.root.join("agent-settings.json");
        if main_settings.is_file() { fs::copy(main_settings, profile.join("agent-settings.json"))?; }
        roster.bots.push(bot.clone());
        atomic_json(&self.roster_path(), &roster)?;
        Ok(bot)
    }

    pub fn update_bot(&self, name: &str, display_name: &str, description: &str, emoji: &str) -> Result<Bot> {
        let mut roster = self.read_roster()?;
        let bot = roster.bots.iter_mut().find(|bot| bot.name == name).context("This bot no longer exists")?;
        ensure!(!display_name.trim().is_empty(), "Give the bot a name");
        bot.display_name = display_name.trim().into(); bot.description = description.trim().into(); bot.emoji = emoji.trim().chars().take(4).collect();
        let updated = bot.clone();
        atomic_json(&self.roster_path(), &roster)?;
        Ok(updated)
    }

    pub fn bot_soul(&self, name: &str) -> Result<String> {
        Ok(fs::read_to_string(self.bot_profile(name)?.join("SOUL.md")).unwrap_or_default())
    }
    pub fn set_bot_soul(&self, name: &str, text: &str) -> Result<()> {
        ensure!(self.bots()?.iter().any(|bot| bot.name == name), "This bot no longer exists");
        let profile = self.bot_profile(name)?; fs::create_dir_all(&profile)?;
        let mut file = NamedTempFile::new_in(&profile)?; file.write_all(text.as_bytes())?; file.persist(profile.join("SOUL.md"))?;
        Ok(())
    }

    /// Removes the bot and its profile. Its chat is archived, never deleted.
    pub fn delete_bot(&self, name: &str) -> Result<()> {
        let mut roster = self.read_roster()?;
        let before = roster.bots.len();
        roster.bots.retain(|bot| bot.name != name);
        ensure!(roster.bots.len() < before, "This bot no longer exists");
        atomic_json(&self.roster_path(), &roster)?;
        self.with_catalog(true, |catalog| {
            for record in catalog.sessions.iter_mut().filter(|r| r.bot.as_deref() == Some(name)) { record.archived = true; record.bot = None; record.title = format!("{} (deleted bot)", record.title); }
            Ok(())
        })?;
        let profile = self.bot_profile(name)?;
        if profile.is_dir() { fs::remove_dir_all(profile)?; }
        Ok(())
    }

    /// Resolve the registry every time: the session titled "Bot Chat" owned by
    /// this bot. Created (adopt-before-mint, under the catalog lock) when missing.
    pub fn bot_chat(&self, name: &str) -> Result<SessionSummary> {
        ensure!(self.bots()?.iter().any(|bot| bot.name == name), "This bot no longer exists");
        self.with_catalog(true, |catalog| {
            if let Some(record) = catalog.sessions.iter_mut().find(|r| r.bot.as_deref() == Some(name) && r.title == BOT_CHAT_TITLE) {
                record.archived = false;
                let summary = self.summary(record);
                catalog.active_session = Some(summary.id.clone());
                return Ok(summary);
            }
            let mut record = self.new_record(None, Some(BOT_CHAT_TITLE), SessionSource::Native);
            record.bot = Some(name.to_owned());
            let directory = self.create_directory(&record.id)?;
            if let Err(error) = atomic_json(&directory.join("conversation.json"), &History::default()) {
                let _ = fs::remove_dir(&directory);
                return Err(error);
            }
            catalog.active_session = Some(record.id.clone());
            let summary = self.summary(&record);
            catalog.sessions.push(record);
            Ok(summary)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bot_owns_one_hidden_canonical_chat() {
        let dir = tempfile::tempdir().unwrap();
        let library = Library::open(dir.path()).unwrap();
        let bot = library.create_bot("Research Helper", "Finds papers.", "Cite sources.", "🔬").unwrap();
        assert_eq!(bot.name, "research-helper");
        assert!(library.bot_soul(&bot.name).unwrap().contains("Cite sources."));
        assert!(library.create_bot("research helper", "", "", "").is_err(), "names are unique");
        let chat = library.bot_chat(&bot.name).unwrap();
        assert_eq!(chat.title, BOT_CHAT_TITLE);
        // Same row every time, even after the user renames other chats or time passes.
        assert_eq!(library.bot_chat(&bot.name).unwrap().id, chat.id);
        // Hidden from the Sessions list, and never reused as a "New chat" draft.
        assert!(library.list_sessions(&SessionQuery::default()).unwrap().is_empty());
        assert_ne!(library.create_session(None, None).unwrap().id, chat.id);
        library.delete_bot(&bot.name).unwrap();
        assert!(library.bots().unwrap().is_empty());
        assert!(!library.bot_profile(&bot.name).unwrap().exists());
        assert!(library.bot_chat(&bot.name).is_err());
    }
}

/// A group chat (plugins/hermes-bots group-chat*): a named room of bots that
/// answer the user in turn. Its chat is a hidden session owned by "group:<id>".
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Group { pub id: String, pub name: String, pub members: Vec<String> }

#[derive(Default, Serialize, Deserialize)]
struct Groups { #[serde(default)] groups: Vec<Group> }

impl Library {
    fn groups_path(&self) -> PathBuf { self.root.join("groups.json") }
    pub fn groups(&self) -> Result<Vec<Group>> {
        match fs::read(self.groups_path()) {
            Ok(bytes) => Ok(serde_json::from_slice::<Groups>(&bytes).context("Could not read group chats")?.groups),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(vec![]),
            Err(e) => Err(e.into()),
        }
    }
    pub fn create_group(&self, name: &str, members: &[String]) -> Result<Group> {
        let name = name.trim();
        ensure!(!name.is_empty(), "Name the group chat");
        let bots = self.bots()?;
        ensure!(members.len() >= 2, "A group chat needs at least two bots");
        ensure!(members.iter().all(|m| bots.iter().any(|b| &b.name == m)), "A selected bot no longer exists");
        let mut all = self.groups()?;
        let group = Group { id: new_id("group"), name: name.into(), members: members.to_vec() };
        all.push(group.clone());
        atomic_json(&self.groups_path(), &Groups { groups: all })?;
        Ok(group)
    }
    pub fn delete_group(&self, id: &str) -> Result<()> {
        let mut all = self.groups()?;
        all.retain(|g| g.id != id);
        atomic_json(&self.groups_path(), &Groups { groups: all })?;
        let owner = format!("group:{id}");
        self.with_catalog(true, |catalog| { for r in catalog.sessions.iter_mut().filter(|r| r.bot.as_deref() == Some(owner.as_str())) { r.archived = true; r.bot = None; } Ok(()) })
    }
    /// The group's one chat, resolved by owner every time (created when missing).
    pub fn group_chat(&self, id: &str) -> Result<SessionSummary> {
        let group = self.groups()?.into_iter().find(|g| g.id == id).context("This group chat no longer exists")?;
        let owner = format!("group:{id}");
        self.with_catalog(true, |catalog| {
            if let Some(record) = catalog.sessions.iter().find(|r| r.bot.as_deref() == Some(owner.as_str())) {
                let summary = self.summary(record); catalog.active_session = Some(summary.id.clone()); return Ok(summary);
            }
            let mut record = self.new_record(None, Some(&group.name), SessionSource::Native);
            record.bot = Some(owner.clone());
            let directory = self.create_directory(&record.id)?;
            if let Err(error) = atomic_json(&directory.join("conversation.json"), &History::default()) { let _ = fs::remove_dir(&directory); return Err(error); }
            catalog.active_session = Some(record.id.clone());
            let summary = self.summary(&record);
            catalog.sessions.push(record);
            Ok(summary)
        })
    }
}

#[cfg(test)]
mod group_tests {
    use super::*;
    #[test]
    fn group_needs_two_bots_and_owns_one_hidden_chat() {
        let dir = tempfile::tempdir().unwrap();
        let library = Library::open(dir.path()).unwrap();
        let a = library.create_bot("Ada", "", "", "").unwrap();
        assert!(library.create_group("Solo", &[a.name.clone()]).is_err());
        let b = library.create_bot("Bo", "", "", "").unwrap();
        let group = library.create_group("Team", &[a.name.clone(), b.name.clone()]).unwrap();
        let chat = library.group_chat(&group.id).unwrap();
        assert_eq!(library.group_chat(&group.id).unwrap().id, chat.id);
        assert!(library.list_sessions(&SessionQuery::default()).unwrap().is_empty(), "group chats stay out of Sessions");
        library.delete_group(&group.id).unwrap();
        assert!(library.groups().unwrap().is_empty());
    }
}
