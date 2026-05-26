//! Lycus Communication Module
//!
//! Provides inter-agent communication over Matrix protocol.
//! Each Lycus agent connects to a shared private homeserver and communicates
//! through dedicated Matrix rooms. Incoming messages trigger the internal cron system.

mod matrix;
mod bridge;
mod config;
mod discovery;

pub use matrix::MatrixClient;
pub use bridge::CronBridge;
pub use config::CommConfig;
pub use discovery::AgentDiscovery;

/// Main entry point for the communication module
pub struct LycusComm {
    matrix_client: MatrixClient,
    cron_bridge: CronBridge,
    discovery: AgentDiscovery,
}

impl LycusComm {
    /// Create a new LycusComm instance
    pub async fn new(config: CommConfig) -> Result<Self, anyhow::Error> {
        let matrix_client = MatrixClient::new(config.matrix_config())?;
        let cron_bridge = CronBridge::new(config.bridge_config())?;
        let discovery = AgentDiscovery::new(config.discovery_config())?;
        
        Ok(Self {
            matrix_client,
            cron_bridge,
            discovery,
        })
    }
    
    /// Start the communication module
    pub async fn start(&self) -> Result<(), anyhow::Error> {
        println!("Starting Lycus communication module...");
        
        // Connect to Matrix homeserver
        self.matrix_client.connect().await?;
        
        // Join the agent room
        self.matrix_client.join_agent_room().await?;
        
        // Start listening for messages
        self.matrix_client.start_listener().await?;
        
        // Start cron bridge
        self.cron_bridge.start()?;
        
        // Register with discovery service
        self.discovery.register_local_agent().await?;
        
        println!("Lycus communication module started successfully");
        Ok(())
    }
    
    /// Send a message to another agent
    pub async fn send_message(&self, target_agent: &str, message: &str) -> Result<(), anyhow::Error> {
        self.matrix_client.send_message(target_agent, message).await
    }
    
    /// Discover available agents
    pub async fn discover_agents(&self) -> Result<Vec<String>, anyhow::Error> {
        self.discovery.list_agents().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_comm_module_creation() {
        // Basic test to ensure module compiles
        assert!(true);
    }
}
