# Setup & Billing

## Getting an API Key

1. Go to **https://cloud.mobilerun.ai/api-keys** (sign in with Google, GitHub, or Discord)
2. Click **"New Key"**, name it, copy the full key — shown only once
3. Key format: `dr_sk_...` — anything without this prefix is not a Mobilerun key

**Auth issues:**
- `401` on a working key → may be revoked, create a new one
- Can't find API keys page → https://cloud.mobilerun.ai/api-keys (must be logged in)
- No account → https://cloud.mobilerun.ai/sign-in (first login creates one)

---

## Connecting a Personal Phone (Android)

1. Download the Portal APK: open Chrome on the phone, go to **https://droidrun.ai/portal**
2. Install the APK (approve sideloading — the app is open source: https://github.com/droidrun/mobilerun-portal)
3. Open the app, tap **"Enable Now"** to grant Accessibility Service
4. Connect:
   - **Login:** tap "Connect to Mobilerun" → opens browser login (Google/GitHub/Discord)
   - **API key:** long-press "Connect to Mobilerun" → paste key from dashboard
5. Device appears in `GET /devices` with `state: "ready"`

**Common issues:**
- `disconnected` → Portal app closed or phone lost network. Reopen Portal.
- Device not appearing → check Accessibility Service is enabled, Portal is open, phone has internet
- Switch accounts → tap Logout in Portal (clears credentials), then Connect again

## Connecting an iPhone

Requires a Mac with Xcode + iPhone via USB. Enable Developer Mode, install WebDriverAgent.
Full guide: https://docs.mobilerun.ai/guides/connect-iphone

---

## Cloud & Physical Phone Setup

Full guides: https://docs.mobilerun.ai/guides/cloud-phone-setup | https://docs.mobilerun.ai/guides/physical-phone-setup

**Before provisioning:**
1. Create a Google account — have credentials ready
2. Configure a SOCKS5 proxy — Cloud and Physical Phones need one for internet (Personal Phones don't)
3. Match proxy country to locale — mismatches are a detection signal. Smart IP auto-aligns GPS/timezone/language.

**After provisioning:**
- Install apps via Google Play Store only — no third-party APK sideloading
- Wait for `ready` state — provisioning takes a few minutes

**eSIM (Physical Phones only):**
All Physical Phones are hosted in Germany. eSIM must support activation/roaming there.

---

## Device Types & Pricing

No base plan required — sign up free, add devices as needed.

| Device | Hardware | Cost | Credits/mo |
|--------|----------|------|-----------|
| Personal Phone | Your own device | $5/mo | 250 |
| Cloud Phone | High-performance virtual | $50/mo | 2,500 |
| Physical Phone | Real hardware in data center | $150/mo | 5,000 |

### Credits

1 credit = $0.01 USD. ~0.5 credits per agent step (varies by model).
Top up: $5 per 500 credits (no expiry). Monitor at https://cloud.mobilerun.ai/billing

---

## Resources

| Resource | URL |
|----------|-----|
| Dashboard | https://cloud.mobilerun.ai |
| API Keys | https://cloud.mobilerun.ai/api-keys |
| Billing | https://cloud.mobilerun.ai/billing |
| Docs | https://docs.mobilerun.ai |
| Framework (GitHub) | https://github.com/droidrun/droidrun |
| Portal App (GitHub) | https://github.com/droidrun/droidrun-portal |
| Discord | https://discord.gg/kc2JYQfX2c |
| YouTube | https://www.youtube.com/@droidrun |
| Python SDK | https://pypi.org/project/droidrun/ |
