# Architecture Diagram Patterns — Worked XML

Full, copy-adaptable `.drawio` XML for the most common architecture requests. All follow the rigid grid (`x = col*180 + 40`, `y = row*120 + 40`; rect `140×60`, cylinder `100×70`) and the well-formedness rules in SKILL.md.

> These are meant to be passed to `open_drawio_xml` (MCP) or written to a `.drawio` file and exported by the CLI. They contain no XML comments — copy them as-is.

---

## 1. Layered Service Architecture (top → bottom)

Layers: Client → API → Logic → Data. Each layer is a titled swimlane; components inside; arrows show downward flow.

```xml
<mxfile host="app.diagrams.net" modified="2026-01-01T00:00:00" agent="agent" version="24.0.0" type="device">
  <diagram name="Layered" id="lay1">
    <mxGraphModel dx="1200" dy="900" grid="1" gridSize="10" page="1" pageWidth="1100" pageHeight="850" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="l_client" value="Clients" style="swimlane;startSize=30;fillColor=#dae8fc;strokeColor=#6c8ebf;html=1;" vertex="1" parent="1">
          <mxGeometry x="40" y="40" width="900" height="100" as="geometry" />
        </mxCell>
        <mxCell id="web" value="Web App" style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="l_client">
          <mxGeometry x="40" y="45" width="140" height="40" as="geometry" />
        </mxCell>
        <mxCell id="mobile" value="Mobile App" style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="l_client">
          <mxGeometry x="220" y="45" width="140" height="40" as="geometry" />
        </mxCell>
        <mxCell id="l_api" value="API Gateway" style="swimlane;startSize=30;fillColor=#ffe6cc;strokeColor=#d79b00;html=1;" vertex="1" parent="1">
          <mxGeometry x="40" y="160" width="900" height="100" as="geometry" />
        </mxCell>
        <mxCell id="gw" value="REST Gateway" style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="l_api">
          <mxGeometry x="40" y="45" width="160" height="40" as="geometry" />
        </mxCell>
        <mxCell id="l_logic" value="Business Logic" style="swimlane;startSize=30;fillColor=#d5e8d4;strokeColor=#82b366;html=1;" vertex="1" parent="1">
          <mxGeometry x="40" y="280" width="900" height="100" as="geometry" />
        </mxCell>
        <mxCell id="svc1" value="Order Service" style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="l_logic">
          <mxGeometry x="40" y="45" width="160" height="40" as="geometry" />
        </mxCell>
        <mxCell id="svc2" value="User Service" style="rounded=1;whiteSpace=wrap;html=1;" vertex="1" parent="l_logic">
          <mxGeometry x="240" y="45" width="160" height="40" as="geometry" />
        </mxCell>
        <mxCell id="l_data" value="Data" style="swimlane;startSize=30;fillColor=#f5f5f5;strokeColor=#666666;html=1;" vertex="1" parent="1">
          <mxGeometry x="40" y="400" width="900" height="120" as="geometry" />
        </mxCell>
        <mxCell id="db" value="PostgreSQL" style="shape=cylinder3;whiteSpace=wrap;html=1;" vertex="1" parent="l_data">
          <mxGeometry x="40" y="45" width="100" height="70" as="geometry" />
        </mxCell>
        <mxCell id="cache" value="Redis" style="shape=cylinder3;whiteSpace=wrap;html=1;" vertex="1" parent="l_data">
          <mxGeometry x="200" y="45" width="100" height="70" as="geometry" />
        </mxCell>
        <mxCell id="e1" value="HTTPS" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;" edge="1" parent="1" source="web" target="gw"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e2" value="calls" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;" edge="1" parent="1" source="gw" target="svc1"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e3" value="reads/writes" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;" edge="1" parent="1" source="svc1" target="db"><mxGeometry relative="1" as="geometry" /></mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

---

## 2. Microservices with an Event Bus (hub pattern)

Place the bus in the **center of the service row** so peers reach it with short horizontal arrows — no crossings.

```xml
<mxfile host="app.diagrams.net" modified="2026-01-01T00:00:00" agent="agent" version="24.0.0" type="device">
  <diagram name="Microservices" id="ms1">
    <mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" page="1" pageWidth="1100" pageHeight="850" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="s1" value="Order Svc" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="40" y="200" width="160" height="60" as="geometry" /></mxCell>
        <mxCell id="s2" value="Payment Svc" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="40" y="360" width="160" height="60" as="geometry" /></mxCell>
        <mxCell id="bus" value="Kafka" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f8cecc;strokeColor=#b85450;fontStyle=1;" vertex="1" parent="1"><mxGeometry x="420" y="280" width="160" height="60" as="geometry" /></mxCell>
        <mxCell id="s3" value="Inventory Svc" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="800" y="200" width="160" height="60" as="geometry" /></mxCell>
        <mxCell id="s4" value="Notify Svc" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="800" y="360" width="160" height="60" as="geometry" /></mxCell>
        <mxCell id="e1" value="pub" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" parent="1" source="s1" target="bus"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e2" value="pub" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" parent="1" source="s2" target="bus"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e3" value="sub" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" parent="1" source="bus" target="s3"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e4" value="sub" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" parent="1" source="bus" target="s4"><mxGeometry relative="1" as="geometry" /></mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

---

## 3. Client → API → Database (left-to-right)

Use `postLayout:"elk"` + `direction:"horizontal"` (MCP) or the rigid grid left-to-right.

```xml
<mxfile host="app.diagrams.net" modified="2026-01-01T00:00:00" agent="agent" version="24.0.0" type="device">
  <diagram name="Client-API-DB" id="cad1">
    <mxGraphModel dx="1200" dy="600" grid="1" gridSize="10" page="1" pageWidth="1100" pageHeight="850" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="client" value="Client" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="40" y="200" width="140" height="60" as="geometry" /></mxCell>
        <mxCell id="api" value="API" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#d79b00;" vertex="1" parent="1"><mxGeometry x="220" y="200" width="140" height="60" as="geometry" /></mxCell>
        <mxCell id="db" value="Database" style="shape=cylinder3;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="1"><mxGeometry x="400" y="195" width="100" height="70" as="geometry" /></mxCell>
        <mxCell id="e1" value="request" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;" edge="1" parent="1" source="client" target="api"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e2" value="query" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;" edge="1" parent="1" source="api" target="db"><mxGeometry relative="1" as="geometry" /></mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

---

## 4. C4-lite System Context

Top-level boxes: Person(s) + System(s) + External dependencies. Color-code by type; add a legend.

```xml
<mxfile host="app.diagrams.net" modified="2026-01-01T00:00:00" agent="agent" version="24.0.0" type="device">
  <diagram name="C4-Context" id="c4c1">
    <mxGraphModel dx="1200" dy="700" grid="1" gridSize="10" page="1" pageWidth="1100" pageHeight="850" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="user" value="Customer" style="ellipse;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="1"><mxGeometry x="40" y="240" width="100" height="80" as="geometry" /></mxCell>
        <mxCell id="sys" value="Internet Banking&#xa;System" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#82b366;fontStyle=1;" vertex="1" parent="1"><mxGeometry x="320" y="220" width="200" height="120" as="geometry" /></mxCell>
        <mxCell id="ext" value="Mainframe&#xa;(legacy)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#666666;" vertex="1" parent="1"><mxGeometry x="680" y="240" width="160" height="80" as="geometry" /></mxCell>
        <mxCell id="e1" value="uses" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;" edge="1" parent="1" source="user" target="sys"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e2" value="replicates" style="edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;dashed=1;" edge="1" parent="1" source="sys" target="ext"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="legend" value="Legend" style="rounded=0;whiteSpace=wrap;html=1;fillColor=none;strokeColor=#666666;verticalAlign=top;fontStyle=1;" vertex="1" parent="1"><mxGeometry x="40" y="420" width="220" height="90" as="geometry" /></mxCell>
        <mxCell id="leg1" value="" style="rounded=0;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;" vertex="1" parent="legend"><mxGeometry x="10" y="28" width="30" height="16" as="geometry" /></mxCell>
        <mxCell id="leg1t" value="Person" style="text;html=1;align=left;verticalAlign=middle;" vertex="1" parent="legend"><mxGeometry x="50" y="26" width="160" height="20" as="geometry" /></mxCell>
        <mxCell id="leg2" value="" style="rounded=0;html=1;fillColor=#d5e8d4;strokeColor=#82b366;" vertex="1" parent="legend"><mxGeometry x="10" y="52" width="30" height="16" as="geometry" /></mxCell>
        <mxCell id="leg2t" value="System" style="text;html=1;align=left;verticalAlign=middle;" vertex="1" parent="legend"><mxGeometry x="50" y="50" width="160" height="20" as="geometry" /></mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

---

## Decision guidance (which pattern)

| User says | Use |
|-----------|-----|
| "layered architecture", "n-tier", "stack" | Pattern 1 |
| "microservices", "event-driven", "pub/sub", "Kafka" | Pattern 2 (hub bus) |
| "request flow", "client to API to DB", "left to right" | Pattern 3 |
| "C4", "system context", "who uses what" | Pattern 4 |
| "VPC/AZ/subnet", "cloud topology" | Nested swimlanes (see SKILL.md "Containers & nested architecture") + cloud icons |
| "flowchart", "decision tree", "process" | Mermaid via `open_drawio_mermaid` or diamond shapes |
| "ER diagram", "schema" | `edgeStyle=entityRelationEdgeStyle`, crow's-foot, cylinders for tables |
