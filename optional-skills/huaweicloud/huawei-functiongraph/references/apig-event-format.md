# APIG DEDICATEDGATEWAY Event Format

When APIG calls FunctionGraph via DEDICATEDGATEWAY trigger, the event structure is NOT the standard HTTP event. Key differences:

## Event Structure

```json
{
  "isBase64Encoded": true,
  "body": "<Base64-encoded request body>",
  "httpMethod": "POST",
  "requestContext": {
    "apiId": "xxx",
    "requestId": "xxx",
    "stage": "RELEASE"
  }
}
```

## Critical: body is Base64 Encoded

`event["body"]` is Base64 encoded. You MUST decode it:

**Python handler template:**

```python
import json, base64

def handler(event, context):
    body = event.get("body", "")
    if event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")
    data = json.loads(body) if body else {}
    return {
        "statusCode": 200,
        "body": json.dumps({"message": "ok", "received": data}),
        "headers": {"Content-Type": "application/json"}
    }
```

**Node.js handler template:**

```js
exports.handler = async (event, context) => {
  let body = event.body || '';
  if (event.isBase64Encoded) {
    body = Buffer.from(body, 'base64').toString('utf8');
  }
  const data = body ? JSON.parse(body) : {};
  return {
    statusCode: 200,
    body: JSON.stringify({ message: 'ok', received: data }),
    headers: { 'Content-Type': 'application/json' },
  };
};
```

## Missing Fields vs Standard HTTP

Fields NOT in DEDICATEDGATEWAY event: `event.path`, `event.headers`, `event.queryStringParameters`. Use `event.httpMethod` for method detection.
