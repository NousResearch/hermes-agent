# Cloud Provider Icons (hand-authored cheatsheet)

For AWS/Azure/GCP/Cisco/Kubernetes, prefer **official icons**. With the MCP server, call `search_shapes "aws lambda"` and paste the returned `style`. This sheet is for hand-authored XML when the MCP is unavailable. draw.io renders icons via `shape=mxgraph.*` (AWS) or `image=img/lib/...svg` (Azure/GCP). Wrong names render as blank boxes.

> With MCP, **always prefer `search_shapes`** over guessing these strings — it returns the exact current `style` and size.

---

## AWS — `shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.SERVICE`

```xml
<mxCell id="lambda-1" value="Lambda" style="sketch=0;outlineConnect=0;fontColor=#232F3E;fillColor=#ED7100;strokeColor=#ffffff;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=12;aspect=fixed;shape=mxgraph.aws4.resourceIcon;resIcon=mxgraph.aws4.lambda;" vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="78" height="78" as="geometry" />
</mxCell>
```

**Common `resIcon` values:**
- Compute: `ec2`, `lambda`, `ecs`, `eks`, `fargate`
- Storage: `s3`, `elastic_block_store`, `elastic_file_system`
- Database: `rds`, `dynamodb`, `aurora`, `elasticache`, `redshift`
- Networking: `vpc`, `cloudfront`, `route_53`, `api_gateway`, `elastic_load_balancing`
- Security: `iam`, `cognito`, `secrets_manager`, `kms`, `waf`

**AWS category colors:** Compute `#ED7100` · Storage `#7AA116` · Database `#C925D1` · Networking `#8C4FFF` · Security `#DD344C`

**AWS group containers:**
```xml
<mxCell id="vpc" value="VPC" style="sketch=0;html=1;whiteSpace=wrap;fontSize=12;shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_vpc;strokeColor=#8C4FFF;fillColor=none;verticalAlign=top;align=left;spacingLeft=30;" vertex="1" parent="1">
  <mxGeometry x="40" y="40" width="400" height="300" as="geometry" />
</mxCell>
```

Subnet variants reuse the same group shape with a different `grIcon` and colors:
- **Public Subnet:** `grIcon=mxgraph.aws4.group_public_subnet; strokeColor=#7AA116; fillColor=#E9F3E6`
- **Private Subnet:** `grIcon=mxgraph.aws4.group_private_subnet; strokeColor=#00A4A6; fillColor=#E6F6F7`
```

---

## Azure — `image=img/lib/azure2/CATEGORY/SERVICE.svg`

```xml
<mxCell id="vm-1" value="VM" style="aspect=fixed;html=1;align=center;image;fontSize=12;image=img/lib/azure2/compute/Virtual_Machine.svg;verticalLabelPosition=bottom;verticalAlign=top;" vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="68" height="68" as="geometry" />
</mxCell>
```

**Common image paths:**
- `compute/Virtual_Machine.svg`, `compute/Function_Apps.svg`
- `containers/Kubernetes_Services.svg`
- `storage/Storage_Accounts.svg`, `storage/Blob_Storage.svg`
- `databases/SQL_Database.svg`, `databases/Azure_Cosmos_DB.svg`
- `networking/Virtual_Networks.svg`, `networking/Load_Balancers.svg`
- `security/Key_Vaults.svg`, `identity/Azure_Active_Directory.svg`

---

## GCP — `image=img/lib/azure2/...` does NOT apply; use `image=img/lib/gcp/...`

GCP icons live under `img/lib/gcp/`. Pattern mirrors Azure:
```xml
<mxCell id="gce-1" value="Compute Engine" style="aspect=fixed;html=1;align=center;image;fontSize=12;image=img/lib/gcp/compute/ComputeEngine.svg;verticalLabelPosition=bottom;verticalAlign=top;" vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="68" height="68" as="geometry" />
</mxCell>
```
Common: `compute/ComputeEngine.svg`, `storage/CloudStorage.svg`, `bigdata/BigQuery.svg`, `network/Vpc.svg`, `containers/KubernetesEngine.svg`.

> GCP path correctness varies by draw.io version — when in doubt, use `search_shapes "gcp <service>"`.

---

## Brand / concept logos (no cloud prefix)

`search_shapes` returns these as `shape=image;...` styles from the draw.io icon service (e.g. `react`, `slack`, `shopping cart`, `kubernetes`). For a known public URL you can also inline:
```xml
<mxCell id="logo" value="React" style="shape=image;html=1;verticalLabelPosition=bottom;verticalAlign=top;image=https://www.draw.io/images/logo.svg;aspect=fixed;" vertex="1" parent="1">
  <mxGeometry x="100" y="100" width="60" height="60" as="geometry" />
</mxCell>
```
For AI/LLM brand logos (OpenAI, Claude, Gemini…) which draw.io has none of, the `Agents365-ai/drawio-skill` ships `scripts/aiicons.py "<brand>"` returning a CDN `image` style.

---

## Cisco / network / Kubernetes / P&ID / electrical / BPMN

These libraries are large and version-sensitive. **Do not guess** — call `search_shapes "cisco router"`, `search_shapes "kubernetes pod"`, `search_shapes "bpmn task"`, etc., and paste the returned `style` verbatim.
