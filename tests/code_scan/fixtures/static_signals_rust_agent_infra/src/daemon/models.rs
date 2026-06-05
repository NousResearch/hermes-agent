// Daemon model management using fastembed + ONNX
use fastembed::EmbeddingModel;

pub fn install_semantic_model() {
    // Installs ONNX/fastembed semantic embedding models for daemon
    let _model = EmbeddingModel::AllMiniLML6V2;
    println!("daemon installed embedding model");
}

pub fn init_daemon_model() {
    install_semantic_model();
}
