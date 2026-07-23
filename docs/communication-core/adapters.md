# Adapter capability matrix

Capabilities are fail-closed. Declaring a capability without implementing its
method raises `CapabilityUnsupportedError`; an unconfigured connector reports
failed health and an empty capability list.

| Adapter | Contacts | Profiles | Conversations | Messages | Groups | Events | Receipts | Send |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Facebook local CRM bridge | read | read | read | read | unsupported | read | unsupported | forbidden |
| Telegram Communication, configured connector | connector-declared | connector-declared | connector-declared | connector-declared | connector-declared | connector-declared | connector-declared | forbidden |
| Telegram, no connector | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported | forbidden |
| VK, configured connector/test server | connector-declared | connector-declared | connector-declared | connector-declared | connector-declared | connector-declared | connector-declared | forbidden |
| VK, no connector | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported | unsupported | forbidden |
| Dating pilot | blocked until a named user-confirmed provider/test account exists | blocked | blocked | blocked | blocked | blocked | blocked | forbidden |
| Fake test adapter | read | read | read | read | read | read | read | fake sink only |

Facebook reuses the canonical Facebook repository and opens legacy storage
read-only; it does not own browser synchronization. Telegram/VK share the
`CommunicationReadConnector` protocol and require an injected real connector
or deterministic fixture. News ingest is not a Telegram Communication adapter
and never creates `Person` records.
