use std::env;
use std::process;

fn main() {
    let wants_health = env::args().skip(1).any(|arg| arg == "--health");

    if wants_health {
        let payload = hermes_proto::health_response("health-check");
        println!("{}", serde_json::to_string(&payload).unwrap());
        return;
    }

    eprintln!("usage: hermes-sidecar --health");
    process::exit(2);
}

#[cfg(test)]
mod tests {
    #[test]
    fn health_response_is_serializable() {
        let payload = hermes_proto::health_response("unit-test");
        let json = serde_json::to_string(&payload).unwrap();
        assert!(json.contains("\"service\":\"hermes-sidecar\""));
        assert!(json.contains("\"status\":\"ok\""));
    }
}
