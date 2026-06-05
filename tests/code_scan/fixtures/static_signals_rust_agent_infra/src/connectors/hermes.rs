// Hermes connector for multi-agent integration
use franken_agent_detection::FrakenDetector;

pub struct HermesConnector {
    detector: FrakenDetector,
}

impl HermesConnector {
    pub fn new() -> Self {
        Self { detector: FrakenDetector::default() }
    }

    pub fn detect_franken_agent(&self) -> bool {
        self.detector.is_franken()
    }
}
