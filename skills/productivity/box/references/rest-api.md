# REST API fallback

Use `box request` to extend the CLI when it has no dedicated subcommand. It reuses the configured Box identity, so continue ordinary requested work without asking the user to choose a REST fallback. Confirm only for deletes, access or identity changes, broad or costly batches, or an ambiguous target or scope. Use direct REST only when the CLI is unavailable or application code needs a raw endpoint that an SDK cannot cover.

Using REST does not bypass Box metadata safety rules: inspect metadata instances and existing schemas first, never create or change a metadata template, and retrieve and compare the metadata instance after every write. Never use a file description as an implicit metadata fallback.

## CLI request escape hatch

```bash
box request /files/<FILE_ID> --json
box request /files/<FILE_ID> -X PUT --body '{"name":"renamed.pdf"}' --json
box request /folders -X POST --body '{"name":"New folder","parent":{"id":"0"}}' --json
```

## Create a native Box Note

When asked to create a Box Note, create the native note with the Box Notes API; do not upload plain text with a `.boxnote` suffix. Use the intended parent folder (use `0` only when the user's target is unambiguously their root), then fetch the returned file to verify it:

```bash
box request /notes/convert -X POST \
  --header "box-version: 2026.0" \
  --body '{"content":"# Hello world\n\nhello world","content_format":"markdown","parent":{"id":"0"},"name":"hello-world"}' \
  --json
box files:get <RETURNED_FILE_ID> --json --fields id,name,type,parent
```

`content` is Markdown and is limited to 1 MB. Report the returned file ID and its normal Box file link.

## OAuth identity boundary

`box request` uses the selected OAuth CLI environment. It does not bypass that user's Box permissions. If the CLI is unavailable, use an OAuth-authorized SDK client as described in [SDK development](sdk-development.md). Never echo, log, or commit OAuth tokens or client secrets.

## Sources

- [Box API reference](https://developer.box.com/reference/)
- [Box Notes API: create a note from Markdown](https://developer.box.com/guides/box-notes/convert-markdown/)
- [OAuth 2.0](https://developer.box.com/guides/authentication/oauth2/)
