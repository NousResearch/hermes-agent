# Troubleshooting

Read this when device actions fail, API calls return unexpected errors, or the user reports problems with their phone or the Portal app.

**1. Xiaomi/MIUI kills Accessibility Service (battery optimization)**

- **Symptom:** Device was working, then all actions fail. Accessibility Service appears disabled on the phone.
- **Cause:** Xiaomi/MIUI aggressively kills background services to save battery, including the Droidrun Portal accessibility service.
- **Fix:** Ask the user to:
  1. Open **Settings > Apps > Droidrun Portal > Battery Saver > No restrictions**
  2. Also go to **Settings > Battery > App Battery Saver** and set Droidrun Portal to "No restrictions"
  3. Re-enable the Accessibility Service and reconnect

**2. Device goes to sleep -- shows as `disconnected`**

- **Symptom:** Device was `ready`, now shows `disconnected` after some idle time.
- **Cause:** The phone went to sleep or the OS killed the Portal app due to battery optimization or inactivity.
- **Fix:** Ask the user to wake the phone, reopen the Portal app, and tap Connect. To prevent this:
  - Disable battery optimization for Droidrun Portal
  - Set a longer screen timeout in **Settings > Display > Screen timeout**

**3. IME/Keyboard not active -- `POST /keyboard` fails with 'IME not active'**

- **Symptom:** `POST /devices/{deviceId}/keyboard` returns an error mentioning "IME not active".
- **Cause:** No text input field is currently focused on the device, so there's nowhere to type.
- **Fix:** Before typing, tap on an editable text field first:
  1. Use `GET /devices/{deviceId}/ui-state` to find an element with `"isEditable": true`
  2. Tap that element using `POST /devices/{deviceId}/tap`
  3. Verify the keyboard appeared: check `"keyboardVisible": true` in `GET /devices/{deviceId}/ui-state` > `phone_state`
  4. Now retry `POST /devices/{deviceId}/keyboard`

**4. Task API returns 402 Insufficient Balance**

- **Symptom:** `POST /tasks` returns `402` with "Insufficient Balance" or similar.
- **Cause:** The user's credit balance is too low to run the task.
- **Fix:** Tell the user to add credits or upgrade their plan at https://cloud.mobilerun.ai/billing. To check current balance:
  ```
  GET /user/me
  ```

**5. Screenshot returns 500 Internal Server Error**

- **Symptom:** `GET /devices/{deviceId}/screenshot` returns `500`.
- **Cause:** The device may be in a bad state -- screen off, device locked, Portal app crashed, or a transient server-side error.
- **Fix:**
  1. Verify the device is still `ready`: `GET /devices/{deviceId}`
  2. If `disconnected`, ask the user to reconnect
  3. If still `ready`, retry once -- transient 500s can happen
  4. If it persists, ask the user to reopen the Portal app and check that the phone screen is on and unlocked

**6. Device shows `ready` but all actions fail (ADB connection lost)**

- **Symptom:** Device state is `ready` but taps, screenshots, and other actions all return errors.
- **Cause:** The internal ADB connection between the Portal app and the Mobilerun backend was lost while the device state hasn't updated yet.
- **Fix:**
  1. Ask the user to close and reopen the Droidrun Portal app
  2. Tap Connect again in the Portal app
  3. Wait a few seconds, then re-check with `GET /devices/{deviceId}`
  4. Once the device is back to `ready`, retry the action

**7. Play Store won't open via `PUT /apps/com.android.vending`**

- **Symptom:** Attempting to open the Play Store via the apps endpoint fails or the app doesn't launch.
- **Cause:** Some devices restrict programmatic launching of the Play Store, or the Play Store package name doesn't respond to standard launch intents.
- **Fix:** Use a tap-based approach instead:
  1. Take a screenshot to see the current screen
  2. Navigate to the home screen: `POST /devices/{deviceId}/key` with `HOME`
  3. Look for the Play Store icon in the UI tree and tap it, or use the shell endpoint:
     ```
     POST /devices/{deviceId}/shell
     {"command": "am start -a android.intent.action.MAIN -n com.android.vending/.AssetBrowserActivity"}
     ```

**8. App install from library fails (app not found in library)**

- **Symptom:** Trying to install an app via the apps endpoint returns an error indicating the app isn't found in the library.
- **Cause:** The app hasn't been added to the user's Mobilerun app library. The install endpoint only works with apps that are pre-uploaded to the library.
- **Fix:** Tell the user to:
  1. Add the app to their library at https://cloud.mobilerun.ai (app library section)
  2. Alternatively, install the app manually on the device through the Play Store, then use it directly
  3. For cloud devices, specify required apps in the `apps` array when provisioning: `POST /devices?deviceType=... { "apps": ["com.example.app"] }`

**9. `DELETE /keyboard` (clear) fails -- 'Accessibility fallback failed'**

- **Symptom:** `DELETE /devices/{deviceId}/keyboard` returns an error about accessibility fallback failing.
- **Cause:** The clear-text action relies on the accessibility service to select and delete text. It can fail if the focused field doesn't support text selection or the accessibility service lost its connection.
- **Fix:** Use an alternative approach to clear text:
  1. Type replacement text with the `clearFirst` flag:
     ```
     POST /devices/{deviceId}/keyboard
     {"text": "", "clearFirst": true}
     ```
  2. If that also fails, tap on a different field and then back to reset the focus, then retry
  3. As a last resort, ask the user to check that the Accessibility Service is still enabled on the device

**10. Device in `migrating` state -- what it means and how long to wait**

- **Symptom:** `GET /devices/{deviceId}` returns `state: "migrating"`.
- **Cause:** The cloud device is being moved between physical hosts. This happens during infrastructure maintenance or load balancing and is automatic.
- **Fix:** Wait for migration to complete -- it typically takes 1-5 minutes. Poll the device status with `GET /devices/{deviceId}`. Do not attempt any device actions during migration. Once the state returns to `ready`, resume normal operation. If it stays in `migrating` for more than 10 minutes, suggest the user contact support via Discord (https://discord.gg/kc2JYQfX2c).
