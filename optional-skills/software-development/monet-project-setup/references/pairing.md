# Nearby private transfer protocol

Private transfer is an advanced, user-requested second act after a secret-free
Project Primer has been imported. It is not part of `.monetproj` and does not
make the package sensitive. Do not offer it during the normal handoff; Monet's
native connector checklist is the default path.

## Preconditions

- Hermes runs interactively on the user's local macOS or Windows computer.
- The iPad and computer share a trusted local network.
- `monet-pair` is installed from a signed Monet Desktop release or a trusted
  Monet source checkout.
- The user approves each source environment-variable name before the command
  runs.

Do not transfer from hosted/cloud Hermes, Linux, a public address, a VPN/tunnel,
or an unattended task. Use Monet's setup checklist instead.

## User contract

1. State the Primer slot label, purpose, and environment-variable name. Never
   print or inspect the value.
2. Ask for explicit approval before starting the server.
3. Start the one-time transfer with `monet-pair`; never construct its link
   manually. Present QR scanning only as one way to open that link on iPad.
4. Ask the user to compare the four-digit short authentication string on both
   devices. Stop on a mismatch.
5. The iPad starts with no values selected. The user chooses each value to
   claim, and the source session closes after that one encrypted response.

## Security properties

- QR/link: local endpoint, expiring session ID, project slug, source public
  key. No credential values.
- Key agreement: ephemeral X25519.
- Key separation: HKDF-SHA256 labels for hello, SAS, claim, and secrets.
- Encryption: AES-256-GCM with transcript-bound associated data.
- Authentication: matching four-digit SAS plus authenticated server hello.
- Lifetime: 30-600 seconds, one client, one successful claim.
- Destination: approved values go directly into device-only Keychain slots.
- Logging: request paths, client addresses, values, and encrypted bodies are
  not logged by the pairing server.

The four-digit code is not an encryption key. Do not describe this as
"PIN-encrypted" and do not persist or reuse it.
