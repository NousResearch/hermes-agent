# Greenfield tools reference

Greenfield tools operate on **testnet** or **mainnet**. Use **`network`**: `"testnet"` (default) or `"mainnet"`. Writes require **PRIVATE_KEY** in the MCP server env (or passed where supported).

**Note:** Some tools may be named differently in the implementation (e.g. file upload). Use the tool names exposed by the MCP server; below matches the [BNB Chain MCP README](https://github.com/bnb-chain/bnbchain-mcp) and common usage.

---

## Buckets

| Tool | Description | Parameters |
|------|-------------|------------|
| gnfd_list_buckets | List buckets owned by an address | `network`, `address` (optional), `privateKey` (optional for default account) |
| gnfd_get_bucket_info | Bucket details | `network`, `bucketName` |
| gnfd_get_bucket_full_info | Bucket info and quota usage | `network`, `bucketName`, `privateKey` |
| gnfd_create_bucket | **(Write)** Create a bucket | `network`, `privateKey`, `bucketName` |
| gnfd_delete_bucket | **(Write)** Delete a bucket | `network`, `privateKey`, `bucketName` |

**bucketName** defaults to `created-by-bnbchain-mcp` when optional.

---

## Objects and folders

| Tool | Description | Parameters |
|------|-------------|------------|
| gnfd_list_objects | List objects in a bucket | `network`, `bucketName` |
| gnfd_get_object_info | Object details | `network`, `bucketName`, `objectName` |
| gnfd_upload_object / gnfd_create_file | **(Write)** Upload a file to a bucket | `network`, `privateKey`, `filePath` (absolute path to file), `bucketName` |
| gnfd_download_object | Download object to disk | `network`, `bucketName`, `objectName`, `targetPath` (optional), `privateKey` |
| gnfd_delete_object | **(Write)** Delete an object | `network`, `privateKey`, `bucketName`, `objectName` |
| gnfd_create_folder | **(Write)** Create a folder in a bucket | `network`, `privateKey`, `bucketName`, `folderName` (optional) |

If the server exposes **gnfd_create_file** instead of **gnfd_upload_object**, use **filePath** (absolute path to the file to upload).

---

## Account and payment

| Tool | Description | Parameters |
|------|-------------|------------|
| gnfd_get_account_balance | Balance for a Greenfield account | `network`, address/privateKey as per implementation |
| gnfd_get_payment_accounts | Payment accounts for an address | `network`, `address` (optional), `privateKey` |
| gnfd_get_payment_account_info | Details of a payment account | `network`, account identifier (see implementation) |
| gnfd_create_payment | **(Write)** Create a payment account | `network`, `privateKey` |
| gnfd_get_payment_balance | Payment account balance | `network`, account identifier |
| gnfd_deposit_to_payment | **(Write)** Deposit into payment account | `network`, `to` (payment account address), `amount` (string, in BNB), `privateKey` |
| gnfd_withdraw_from_payment | **(Write)** Withdraw from payment account | `network`, `from`, `amount`, `privateKey` |
| gnfd_disable_refund | **(Write, IRREVERSIBLE)** Disable refund for payment account | `network`, `address`, `privateKey` (if exposed) |

Use **get_supported_networks** or the MCP server’s tool list to confirm exact parameter names for the version in use.
