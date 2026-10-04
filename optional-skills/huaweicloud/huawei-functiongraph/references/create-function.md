# FunctionGraph Function Creation Reference

## Runtime Options

| Runtime        | Handler Format                             | Example                                                           |
| -------------- | ------------------------------------------ | ----------------------------------------------------------------- |
| Python3.9      | `index.handler`                            | `def handler(event, context): return {"statusCode": 200}`         |
| Python3.10     | `index.handler`                            | Same as 3.9                                                       |
| Node.js 18     | `index.handler`                            | `exports.handler = async (event, context) => ({statusCode: 200})` |
| Java 8/11/17   | `com.example.Handler::handleRequest`       | Java class method                                                 |
| Go 1.x         | `handler`                                  | Go function name                                                  |
| C# (.NET Core) | `CsharpDemo::CsharpDemo.Function::Handler` | Namespace.Class::Method                                           |

## Create Function (inline)

```bash
hcloud FunctionGraph CreateFunction \
  --func_name=<name> \
  --package=default \
  --runtime=Python3.10 \
  --handler=index.handler \
  --memory_size=128 \
  --timeout=3 \
  --code_type=inline \
  --func_code.file=<base64-encoded-code> \
  --cli-region=<region>
```

| Param           | Required | Range/Default             |
| --------------- | -------- | ------------------------- |
| `--func_name`   | Yes      | 1-60 chars                |
| `--runtime`     | Yes      | See runtime options above |
| `--handler`     | Yes      | Per runtime               |
| `--memory_size` | No       | 128-4096 MB, default 128  |
| `--timeout`     | No       | 1-900s, default 3         |

## ZIP Upload Warning

`--code_type=zip --code_filename=xxx.zip` **succeeds even with non-existent file** — function created with empty code. Prefer `code_type=inline` for simple functions. If using zip, verify `code_size > 0` after creation and invoke the function to confirm output.

## Required IAM Permissions

- `functiongraph:function:createFunction` — Create
- `functiongraph:function:list` — List
- `functiongraph:function:getConfig` — Read config
- `functiongraph:function:invoke` — Invoke

Grant `FunctionGraph FullAccess` role or custom policy.

## Error Codes

| Error    | Root Cause -> Fix                                                                                                                           |
| -------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| FSS.0400 | `:latest` suffix on URN. Strip it                                                                                                           |
| FSS.1417 | Missing DEDICATEDGATEWAY params                                                                                                             |
| FSS.0403 | Insufficient IAM permission                                                                                                                 |
| FSS.1020 | Missing `--app_xrole` when binding VPC. Create IAM agency with `trust_domain_name=functiongraph`. See `huawei-iam` skill → Agencies section |

## VPC + Agency Configuration

To access VPC-internal resources from FunctionGraph:

```bash
# 1. Create agency (see huawei-iam skill)
hcloud IAM CreateAgency --agency_name=<name> --trust_domain_name=functiongraph
# 2. Grant role (e.g., VPC Administrator)
hcloud IAM GrantRoleToAgency --agency_name=<name> --role_id=<role-id>
# 3. Use in CreateFunction
hcloud FunctionGraph CreateFunction --func_vpc.vpc_id=<vpc> --func_vpc.subnet_id=<subnet> --app_xrole=<name> ...
```

| QuotaExceeded | Max 10 functions per project per region |
