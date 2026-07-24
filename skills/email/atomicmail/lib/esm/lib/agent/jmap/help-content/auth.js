// Help topic: auth (MCP help / AgentSkill help).
export const helpTopicAuth = `\
# Atomic Mail — Auth flow

Auth is automatic after \`register\` (or when \`credentials.json\` + API key
exist).

1. **Challenge** — \`POST /api/v1/challenge\`, read challenge JWT from
   \`Authorization: Bearer <challengeJWT>\`
2. **Proof-of-work** — scrypt until difficulty satisfied
3. **Session JWT** — \`POST /api/v1/session\` with challenge JWT in
   \`Authorization: Bearer ...\` and PoW fields (\`powHex\`, \`nonce\`) in JSON
   body; read session JWT from response \`Authorization: Bearer ...\` (1h TTL);
   signup returns \`apiKey\` once
4. **Capability JWT** — \`POST /api/v1/capability\` with session JWT in
   \`Authorization: Bearer ...\`; read capability JWT from response
   \`Authorization: Bearer ...\` (2 min TTL) used as the JMAP bearer

JWTs are rotated before expiry and written back to disk.

## Credential files (mode 0600)

\`credentials.json\` — \`{ apiKey, inboxId, authUrl, apiUrl, scryptSalt, uploadUrl, downloadUrl }\`  
\`session.jwt\` — session token  
\`capability.jwt\` — capability token

## Overriding defaults

- \`ATOMIC_MAIL_AUTH_URL\` (default: \`https://auth.atomicmail.ai\`)
- \`ATOMIC_MAIL_API_URL\` (default: \`https://api.atomicmail.ai\`)
- \`ATOMIC_MAIL_SCRYPT_SALT\` (optional)
- \`ATOMIC_MAIL_API_KEY\` (optional)
- \`ATOMIC_MAIL_CREDENTIALS_DIR\` (default: \`~/.atomicmail\`)`;
