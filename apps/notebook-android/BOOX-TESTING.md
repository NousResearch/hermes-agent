# BOOX tester guide (Mac Swan)

Thank you for testing Hermes Notebook. This is an early debug build: use a
non-critical notebook page and do not enter secrets into screenshots or bug reports.

## 1. Download and install

1. Open [PR #61687](https://github.com/NousResearch/hermes-agent/pull/61687).
2. Open **Checks**, then the latest successful **Notebook Android APK** run.
3. Download the **Hermes-Notebook-debug** artifact.
4. Unzip the artifact and copy `app-debug.apk` to the BOOX device.
5. Open the APK on BOOX and allow installation from that source when prompted.

The debug APK is not a public release and does not update automatically.

## 2. Connect to Hermes

The BOOX app needs two values from the Hermes operator:

- **Endpoint:** a private HTTPS URL ending at the Hermes Notebook adapter.
- **Notebook token:** the matching `NOTEBOOK_INGEST_TOKEN` value.

Open **Settings** in Hermes Notebook and enter both values. Do not post the token
in Discord, GitHub issues, screenshots, or logs.

The recommended tester route is Tailscale Serve, not a public Funnel. Install
Tailscale on BOOX, join the operator-approved tailnet, and use the private
`https://<hermes-node>.<tailnet>.ts.net` URL supplied by the operator.

## 3. First test

1. Write `Hello from BOOX` with the pen.
2. Wait for **Saved offline**.
3. Add the typed note `Reply with the device and handwriting you received.`
4. Tap **Send**.
5. The first send may download the English handwriting model (about 20 MB).
6. Confirm a Hermes reply appears and contains the recognized handwriting.
7. Close the app, reopen it, and confirm the page returns.

## 4. Pen test matrix

Please report the exact BOOX model, firmware/Android version, and pen model, then test:

- Slow cursive and fast print.
- Light and heavy pressure.
- Pen tilt.
- Built-in eraser, if present.
- Palm resting on the screen while writing.
- A stroke that leaves and re-enters the canvas edge.
- Undo, New page cancellation, and confirmed New page.
- Portrait and landscape rotation.
- Wi-Fi disabled while writing, followed by app restart.
- Send after Wi-Fi returns.
- A page with at least five minutes of handwriting.

## 5. What to send back

For every defect, include:

- BOOX model and firmware version.
- Exact action that triggered it.
- Whether the saved page survived restart.
- Whether the visible ink, recognized text, or Hermes reply was wrong.
- A screenshot or short screen recording when safe.
- Approximate pen delay: instant, noticeable, or unusable.
- Any screen ghosting or flashing.

Never include the endpoint token. A crash report is useful, but review it for
private notebook text before sharing.

## Hermes operator setup

These steps run on the Hermes host, not on the BOOX tablet.

1. Check out the PR branch and install Hermes normally.
2. Enable the bundled platform:

   ```powershell
   hermes plugins enable platforms/kindle
   ```

3. Put a strong random secret in the Hermes profile's `.env`:

   ```text
   NOTEBOOK_INGEST_TOKEN=<random high-entropy secret>
   KINDLE_ALLOWED_USERS=notebook-user
   ```

4. Configure the tools the tester may reach:

   ```yaml
   platform_toolsets:
     kindle:
       - memory
       - live-page
       - browser
       - code_execution
   ```

5. Start/restart the Hermes Gateway and verify locally:

   ```powershell
   Invoke-WebRequest http://127.0.0.1:8793/health -UseBasicParsing
   ```

6. Expose the adapter privately to the tailnet with HTTPS:

   ```powershell
   tailscale serve --bg --https=443 http://127.0.0.1:8793
   tailscale serve status
   ```

7. Give Mac the printed private HTTPS URL and token through a private channel.
   Grant only his BOOX identity/device access in the Tailscale policy. Do not use
   Tailscale Funnel and do not open port 8793 on the router or firewall.

To remove the private proxy after testing:

```powershell
tailscale serve reset
```
