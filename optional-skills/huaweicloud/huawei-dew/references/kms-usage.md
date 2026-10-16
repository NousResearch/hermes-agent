# KMS Usage Guide

## Create Key

hcloud KMS CreateKey --key_alias=app-encryption-key --key_description="Application data encryption"

## List Keys (metadata only)

hcloud KMS ListKeys

## Encrypt (offline preferred)

hcloud KMS EncryptData --key_id=<key-id> --plaintext=<base64-data>

## Decrypt (BLOCKED - use runtime)

DO NOT use hcloud KMS DecryptData directly.
Use SDK with KMS client in application runtime.

## Key Rotation

- Enable: hcloud KMS EnableKeyRotation --key_id=<key-id>
- Rotation period: 365 days (default)

## Key Deletion

- Schedule: hcloud KMS ScheduleKeyDeletion --key_id=<key-id> --pending_days=7
- Cancel during pending window: hcloud KMS CancelKeyDeletion --key_id=<key-id>
- WARNING: After deletion window passes, data encrypted with this key is UNRECOVERABLE
