use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub struct ResponseStoreBackend {
    conn: Connection,
    max_size: usize,
}

impl ResponseStoreBackend {
    pub fn new(max_size: usize, db_path: Option<&str>) -> Result<Self, StoreError> {
        let requested_path = db_path.unwrap_or(":memory:");
        let conn = Connection::open(requested_path).or_else(|_| Connection::open_in_memory())?;
        let _ = conn.pragma_update(None, "journal_mode", "WAL");
        conn.execute(
            "CREATE TABLE IF NOT EXISTS responses (
                response_id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                accessed_at REAL NOT NULL
            )",
            [],
        )?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS conversations (
                name TEXT PRIMARY KEY,
                response_id TEXT NOT NULL
            )",
            [],
        )?;

        Ok(Self { conn, max_size })
    }

    pub fn get(&self, response_id: &str) -> Result<Option<Value>, StoreError> {
        let row: Option<String> = self
            .conn
            .query_row(
                "SELECT data FROM responses WHERE response_id = ?",
                (response_id,),
                |row| row.get(0),
            )
            .optional()?;
        let Some(data) = row else {
            return Ok(None);
        };

        self.conn.execute(
            "UPDATE responses SET accessed_at = ? WHERE response_id = ?",
            params![now_ts(), response_id],
        )?;

        Ok(Some(serde_json::from_str(&data)?))
    }

    pub fn put(&self, response_id: &str, data: &Value) -> Result<(), StoreError> {
        let payload = serde_json::to_string(data)?;
        self.conn.execute(
            "INSERT OR REPLACE INTO responses (response_id, data, accessed_at) VALUES (?, ?, ?)",
            params![response_id, payload, now_ts()],
        )?;

        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM responses", [], |row| row.get(0))?;
        if count > self.max_size as i64 {
            self.conn.execute(
                "DELETE FROM responses WHERE response_id IN (
                    SELECT response_id FROM responses ORDER BY accessed_at ASC LIMIT ?
                )",
                params![count - self.max_size as i64],
            )?;
        }

        Ok(())
    }

    pub fn delete(&self, response_id: &str) -> Result<bool, StoreError> {
        let deleted = self
            .conn
            .execute("DELETE FROM responses WHERE response_id = ?", (response_id,))?;
        Ok(deleted > 0)
    }

    pub fn get_conversation(&self, name: &str) -> Result<Option<String>, StoreError> {
        self.conn
            .query_row(
                "SELECT response_id FROM conversations WHERE name = ?",
                (name,),
                |row| row.get(0),
            )
            .optional()
            .map_err(StoreError::from)
    }

    pub fn set_conversation(&self, name: &str, response_id: &str) -> Result<(), StoreError> {
        self.conn.execute(
            "INSERT OR REPLACE INTO conversations (name, response_id) VALUES (?, ?)",
            params![name, response_id],
        )?;
        Ok(())
    }

    pub fn len(&self) -> Result<usize, StoreError> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM responses", [], |row| row.get(0))?;
        Ok(count as usize)
    }
}

fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn put_get_delete_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("store.db");
        let store = ResponseStoreBackend::new(10, path.to_str()).unwrap();

        store.put("resp_1", &json!({"output": "hello"})).unwrap();
        assert_eq!(store.get("resp_1").unwrap(), Some(json!({"output": "hello"})));
        assert!(store.delete("resp_1").unwrap());
        assert_eq!(store.get("resp_1").unwrap(), None);
    }

    #[test]
    fn lru_eviction_and_access_refresh_match_python_behavior() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("store.db");
        let store = ResponseStoreBackend::new(3, path.to_str()).unwrap();

        store.put("resp_1", &json!({"output": "one"})).unwrap();
        store.put("resp_2", &json!({"output": "two"})).unwrap();
        store.put("resp_3", &json!({"output": "three"})).unwrap();
        let _ = store.get("resp_1").unwrap();
        store.put("resp_4", &json!({"output": "four"})).unwrap();

        assert_eq!(store.get("resp_2").unwrap(), None);
        assert!(store.get("resp_1").unwrap().is_some());
        assert_eq!(store.len().unwrap(), 3);
    }

    #[test]
    fn conversation_mapping_round_trip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("store.db");
        let store = ResponseStoreBackend::new(10, path.to_str()).unwrap();

        store.set_conversation("chat-a", "resp_123").unwrap();
        assert_eq!(
            store.get_conversation("chat-a").unwrap(),
            Some("resp_123".to_string())
        );
    }
}
