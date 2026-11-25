export function spectrumStartupError(error, { localMode }) {
  const detail = error?.stack || String(error);
  let guidance = "Spectrum could not start. Check the original error below.";
  if (localMode && /failed to open database|unable to open database file|SQLITE_CANTOPEN/i.test(detail)) {
    guidance =
      "Local iMessage could not open the Messages database. Check that this Mac is signed into " +
      "Messages and ~/Library/Messages/chat.db exists. In System Settings > Privacy & Security > " +
      "Full Disk Access, authorize /usr/bin/osascript for Hermes's current macOS background " +
      "gateway service, or the terminal app for a manual launch (custom launchers may differ). " +
      "The osascript grant is not Hermes-only. Run `hermes gateway restart` after granting access. " +
      "Terminal access does not verify background-service access.";
  } else if (/ERR_MODULE_NOT_FOUND|MODULE_NOT_FOUND|Cannot find (?:package|module)|NODE_MODULE_VERSION|Could not locate the bindings file/i.test(detail)) {
    guidance = "Spectrum dependencies could not be loaded. Run `hermes photon install-sidecar`.";
  }
  return `photon-sidecar: ${guidance} Original error: ${detail}`;
}
