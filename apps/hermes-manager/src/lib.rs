//! Library modules for the Hermes install manager.

pub mod bundled_manifest;
pub mod commands;
pub mod error;
pub mod installed_manifest;
pub mod ownership;
pub mod paths;
pub mod platform;

pub use error::{ManagerError, Result};
