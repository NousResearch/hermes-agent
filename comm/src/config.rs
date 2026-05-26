//! Configuration for the communication module

use serde::{Deserialize, Serialize};

/// Main configuration for Lycus communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommConfig {
    /// Matrix homeserver URL
    pub homeserver_url: String,
    
    /// Agent username
    pub username: String,
    
    /// Agent password or access token
    pub password: String,
    
    /// Device ID for Matrix client
    pub device_id: String,
    
    /// Name of the shared agent room
    pub agent_room: String,
    
    /// Bridge configuration for cron integration
    pub bridge: BridgeConfig,
    
    /// Discovery service configuration
    pub discovery: DiscoveryConfig,
}

/// Configuration for the cron bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Unix socket path for IPC
    pub socket_path: String,
    
    /// HTTP endpoint for fallback
    pub http_endpoint: String,
    
    /// Message queue size
    pub queue_size: usize,
}

/// Configuration for agent discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enable discovery service
    pub enabled: bool,
    
    /// Discovery interval in seconds
    pub interval_secs: u64,
    
    /// Known agent list (for bootstrap)
    pub known_agents: Vec<String>,
}

impl Default for CommConfig {
    fn default() -> Self {
        Self {
            homeserver_url: "http://localhost:8008".to_string(),
            username: "lycus".to_string(),
            password: "".to_string(),
            device_id: "lycus-comm".to_string(),
            agent_room: "#lycus-agents:lycus.local".to_string(),
            bridge: BridgeConfig::default(),
            discovery: DiscoveryConfig::default(),
        }
    }
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            socket_path: "/tmp/lycus-cron-bridge.sock".to_string(),
            http_endpoint: "http://localhost:8645/trigger".to_string(),
            queue_size: 100,
        }
    }
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_secs: 60,
            known_agents: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = CommConfig::default();
        assert_eq!(config.homeserver_url, "http://localhost:8008");
        assert_eq!(config.bridge.queue_size, 100);
    }
}
