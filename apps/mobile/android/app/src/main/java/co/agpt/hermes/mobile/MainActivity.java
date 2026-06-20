package co.agpt.hermes.mobile;

import android.content.Context;
import android.net.wifi.WifiManager;
import android.os.Build;
import android.os.Bundle;

import com.getcapacitor.BridgeActivity;

/**
 * Holds a high-performance Wi-Fi lock while the app is foregrounded.
 *
 * This app is a thin client to a REMOTE gateway over a long-lived WebSocket.
 * Without the lock, Android Wi-Fi power-saving lets the radio doze between
 * packets — on the lab device this showed as ~90-140ms LAN latency, 3-4s REST
 * round-trips, and the gateway dropping the socket mid-handshake (logged as
 * `send_failed_after_response`), which surfaces in the app as a boot "timeout"
 * even though the gateway is healthy. The lock keeps the radio awake so the
 * connection stays stable; it's released on background to spare the battery.
 */
public class MainActivity extends BridgeActivity {
    private WifiManager.WifiLock wifiLock;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        try {
            WifiManager wifi = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
            if (wifi != null) {
                // LOW_LATENCY (API 29+) is the modern, foreground-only power-save
                // override; fall back to HIGH_PERF on older devices.
                int mode = Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q
                        ? WifiManager.WIFI_MODE_FULL_LOW_LATENCY
                        : WifiManager.WIFI_MODE_FULL_HIGH_PERF;
                wifiLock = wifi.createWifiLock(mode, "hermes:gateway-ws");
                wifiLock.setReferenceCounted(false);
            }
        } catch (Exception ignored) {
            // Wi-Fi lock is a best-effort optimization; never block startup on it.
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        try {
            if (wifiLock != null && !wifiLock.isHeld()) {
                wifiLock.acquire();
            }
        } catch (Exception ignored) {
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        try {
            if (wifiLock != null && wifiLock.isHeld()) {
                wifiLock.release();
            }
        } catch (Exception ignored) {
        }
    }
}
