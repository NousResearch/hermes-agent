# SAP ADT MCP

Hermes integration for reading ABAP source code, CDS views, and repository objects from SAP systems via the [ADT REST API](https://help.sap.com/docs/abap-cloud/abap-development-tools-user-guide/adt-rest-api).

Backed by [adt-mcp](https://github.com/I076453/adt-mcp) — a TypeScript MCP server with 270+ unit tests, surgical `EditSource` diffing, and built-in safety controls.

## What you can do

| Tool | What it does |
|---|---|
| `GetObjectSource` | Read full source of a program, class, function group, etc. |
| `ListPackageContents` | Browse all objects in an ABAP package |
| `SearchObjects` | Find objects by name pattern across the system |
| `GetObjectProperties` | Metadata: object type, package, last changed by, transport |
| `GetIncludes` | List all includes of a program or function group |
| `GetCDSSource` | Read CDS view / BDEF / DCLS source |
| `GetContext` | Compressed public API of a dependency (~80 lines vs 1000+) |
| `RunATC` | Run ABAP Test Cockpit static analysis on an object |
| `RunUnitTests` | Execute ABAP Unit tests with coverage metrics |
| `GetDataPreview` | Preview table or CDS view data (SELECT top N) |

## Prerequisites

- Node.js 18+
- Network access to your SAP system on the ADT port (default 8080)
- An ABAP user with `S_ADT_RES` authorization

## Configuration

`SAP_SYSTEMS` is a JSON array. Each entry describes one system:

```json
[
  {
    "name": "QJ6",
    "baseUrl": "https://your-sap-host:8080",
    "username": "YOUR_USER",
    "password": "YOUR_PASSWORD"
  }
]
```

Store this in `~/.hermes/.env`:

```
SAP_SYSTEMS='[{"name":"QJ6","baseUrl":"https://your-host:8080","username":"user","password":"pass"}]'
SAP_READ_ONLY=true
```

### Optional: restrict to specific packages

```
SAP_ALLOWED_PACKAGES=ZRF_*,/EPD/*
```

Wildcards are supported. Leave unset to allow access to all packages.

## Safety

`SAP_READ_ONLY=true` (the default) blocks all write operations at the server level — Hermes cannot create, modify, or delete any object on the SAP system. This is the recommended setting for analysis workflows such as ABAP report-to-RAP conversion.

Only set `SAP_READ_ONLY=false` on a dedicated development system, never on production.

## Usage example

```
You: Read the source of report rf_steuerinfo from QJ6 and list all its includes.

Hermes: [calls GetObjectSource → GetIncludes → analyses structure]
```
