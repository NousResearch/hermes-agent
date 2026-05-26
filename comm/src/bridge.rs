//! Bridge between Matrix messages and Python cron system
//!
//! Uses Unix socket IPC to communicate with the Python cron scheduler.
//! Incoming Matrix messages are converted into cron triggers.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::net::UnixStream;
use tokio::io::{AsyncWriteExt, AsyncReadExt};

use crate::config::BridgeConfig;

/// Message format for cron bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronTrigger {
    /// Source agent ID
    pub source_agent: String,
    
    /// Message content
    pub message: String,
    
    /// Optional skill to load
    pub skill: Option<String>,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Cron bridge for IPC with Python scheduler
pub struct CronBridge {
    config: BridgeConfig,
    socket_path: PathBuf,
}

impl CronBridge {
    /// Create a new cron bridge
    pub fn new(config: BridgeConfig) -> Result<Self> {
        Ok(Self {
            socket_path: config.socket_path.into(),
            config,
        })
    }
    
    /// Start the bridge listener
    pub fn start(&self) -> Result<()> {
        println!("Starting cron bridge at {}", self.socket_path.display());
        
        // TODO: Implement Unix socket server
        // Listen for incoming triggers from Matrix client
        
        Ok(())
    }
    
    /// Send a trigger to the Python cron system
    pub async fn trigger_cron(&self, trigger: CronTrigger) -> Result<()> {
        println!("Triggering cron with message from {}", trigger.source_agent);
        
        // Serialize the trigger
        let payload = serde_json::to_string(&trigger)?;
        
        // Send via Unix socket
        let mut stream = UnixStream::connect(&self.socket_path).await?;
        stream.write_all(payload.as_bytes()).await?;
        
        // Read acknowledgment
        let mut response = String::new();
        stream.read_to_string(&mut response).await?;
        
        println!("Cron bridge response: {}", response);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bridge_creation() {
        let config = BridgeConfig::default();
        let bridge = CronBridge::new(config);
        assert!(bridge.is_ok());
    }
    
    #[test]
    fn test_trigger_serialization() {
        let trigger = CronTrigger {
            source_agent: "test-agent".to_string(),
            message: "Hello".to_string(),
            skill: None,
            timestamp: 1234567890,
        };
        
        let payload = serde_json::to_string(&trigger);
        assert!(payload.is_ok());
        
        let deserialized: CronTrigger = serde_json::from_str(&payload.unwrap()).unwrap();
        assert_eq!(deserialized.source_agent, "test-agent");
    }
}
