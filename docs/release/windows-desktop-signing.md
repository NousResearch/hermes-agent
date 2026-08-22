# Windows Desktop Signing

Official Windows Desktop releases use Azure Trusted Signing through the
protected GitHub environment `desktop-release`. Ordinary local builds remain
unsigned and keep `build.win.signAndEditExecutable` disabled.

## Maintainer setup

Create the `desktop-release` environment and require a Nous release maintainer
to approve deployments. Configure these environment secrets:

- `AZURE_TENANT_ID`
- `AZURE_CLIENT_ID`
- `AZURE_CLIENT_SECRET`

Configure these environment variables:

- `AZURE_TRUSTED_SIGNING_ENDPOINT`
- `AZURE_TRUSTED_SIGNING_ACCOUNT_NAME`
- `AZURE_TRUSTED_SIGNING_CERTIFICATE_PROFILE_NAME`
- `WINDOWS_SIGNING_PUBLISHER_NAME`

`WINDOWS_SIGNING_PUBLISHER_NAME` must exactly match the certificate Common Name.
Grant the Azure service principal the **Trusted Signing Certificate Profile
Signer** role for the Nous Research signing profile.

## Release

1. Update `apps/desktop/package.json` to the intended version.
2. Create and push the matching annotated tag: `desktop-v<version>`.
3. Approve the `desktop-release` environment deployment.
4. Confirm the workflow verifies Authenticode publisher, timestamp, and trust
   chain before it uploads or publishes any artifact.

The workflow publishes only after both `Get-AuthenticodeSignature` and
`signtool verify /pa /all /v` pass for `win-unpacked\Hermes.exe`, the NSIS
installer, and the MSI.

If the controlled GitHub runner fails specifically while extracting
`winCodeSign`, replace the release overlay with an Azure signing
`afterPack`/artifact hook. Keep `forceCodeSigning`, the strict branding gate,
and the same post-build signature verification; never weaken the workflow to
publish unsigned artifacts.
