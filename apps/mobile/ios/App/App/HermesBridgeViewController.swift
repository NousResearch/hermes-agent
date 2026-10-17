import Capacitor

final class HermesBridgeViewController: CAPBridgeViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        bridge?.registerPluginInstance(SshTunnelPlugin())
    }
}
