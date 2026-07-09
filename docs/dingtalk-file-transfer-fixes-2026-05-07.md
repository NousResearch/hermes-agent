# DingTalk File Transfer Fixes (2026-05-07)

## Summary

This change fixes DingTalk gateway handling for inbound documents and native outbound file delivery. It was validated against a real DingTalk Stream Mode bot conversation by receiving `.xlsx`/`.docx` files, parsing cached document previews, and sending a generated Markdown report back as a native DingTalk file attachment.

## Problems Found

- DingTalk `msgtype=file` callbacks carry file metadata in the raw `content` payload, but `dingtalk-stream` does not expose that payload as a first-class SDK media field. The gateway therefore treated file messages as empty text.
- Download codes from raw file payloads were not resolved and cached, so downstream document parsing never saw a local file path.
- Spreadsheet and Word documents were cached but not summarized into the agent context. The model could see that a file existed, but not its useful contents.
- `send_message` treated DingTalk as a non-media platform and either dropped `MEDIA:` attachments or fell back to webhook-only text delivery.
- `DingTalkAdapter.send_document()` was missing. Sending a local file to a DingTalk conversation requires DingTalk conversation-file and storage OpenAPI calls, not the short-lived reply webhook.
- The DingTalk storage upload API requires `protocol=HEADER_SIGNATURE` and `storageDriver=DINGTALK`; using HTTP method names such as `PUT` fails with `Invalidprotocol`.
- The `MEDIA:` parser returns `(path, is_voice)` tuples. The DingTalk send path must pass only the path into `send_document()`.

## Fixes Made

### Inbound Files

- Preserve raw DingTalk callback payloads on incoming `ChatbotMessage` objects.
- Preserve raw `msgtype=file` `content` as `file_content`.
- Resolve `file_content.downloadCode` through the DingTalk robot file-download API.
- Cache downloaded files through the existing document cache path.
- Map DingTalk file callbacks to `MessageType.DOCUMENT` so the gateway document pipeline runs.

### Document Parsing

- Add lightweight `.xlsx` previews using `openpyxl` when available.
- Add lightweight `.xls` previews using `xlrd` when available.
- Add `.docx` text/table extraction using only stdlib `zipfile` and `xml.etree.ElementTree`.
- Inject bounded document previews into the model context while preserving the cached file path for deeper analysis.

### Outbound Files

- Initialize DingTalk `conv_file_1_0` and `storage_1_0` SDK clients alongside the robot/card clients.
- Add `DingTalkAdapter.send_document()` using this sequence:
  1. `conv_file.get_space(openConversationId, unionId)`
  2. `storage.get_file_upload_info(... protocol=HEADER_SIGNATURE, storageDriver=DINGTALK ...)`
  3. HTTP `PUT` to the signed resource URL
  4. `storage.commit_file(...)`
  5. `conv_file.send(openConversationId, spaceId, dentryId, unionId)`
- Add `DINGTALK_FILE_OPERATOR_UNION_ID` / `DINGTALK_UNION_ID` support for the required operator unionId.
- Add `DINGTALK_FILE_PARENT_ID` support, defaulting to `0`.
- Route DingTalk `MEDIA:` sends through the live gateway adapter instead of webhook text delivery.
- Correctly unwrap `(path, is_voice)` media tuples before calling `send_document()`.

## Required DingTalk App Permissions

The tested file-send path required these DingTalk OpenAPI scopes:

- `ConvFile.Space.Read`
- `Storage.UploadInfo.Read`
- `ConvFile.File.Send`

The app also needs the normal DingTalk Stream/robot credentials already required by the adapter.

## Runtime Configuration

For outbound file sending, configure a real DingTalk unionId for the operator used by the conversation-file APIs:

```env
DINGTALK_FILE_OPERATOR_UNION_ID=<operator-union-id>
# Optional; defaults to 0
DINGTALK_FILE_PARENT_ID=0
```

`senderId` and `senderStaffId` from callbacks are not valid substitutes for unionId. DingTalk returns `paramError-unionId` for those values.

## Verification Performed

- Received `.xlsx` via DingTalk `msgtype=file`, resolved `downloadCode`, cached the workbook locally, and injected a workbook preview into the model context.
- Verified `.docx` extraction with a synthetic Word document.
- Sent `/root/.hermes/profiles/xiaoan/cron/output/incident-rca-audit-report.md` back to the DingTalk conversation as a native file attachment.
- Observed one transient DingTalk `503 ServiceUnavailable`; retry succeeded with DingTalk message id `220382293823`.
- Ran Python bytecode compilation and focused tests for the touched paths.

## Notes

- The short-lived DingTalk reply webhook remains useful for text replies, but it is not enough for native file delivery.
- Optional spreadsheet dependencies degrade gracefully: if `openpyxl` or `xlrd` is unavailable, the gateway still preserves the cached file path and reports that preview parsing was skipped.
