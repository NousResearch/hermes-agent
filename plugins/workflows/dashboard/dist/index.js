(function () {
  "use strict";

  const SDK = window.__HERMES_PLUGIN_SDK__;
  const REG = window.__HERMES_PLUGINS__;
  if (!SDK || !REG || typeof REG.register !== "function" || !SDK.React || !SDK.fetchJSON) return;

  const React = SDK.React;
  const h = React.createElement;
  const Card = SDK.components && SDK.components.Card ? SDK.components.Card : "section";
  const FlowSDK = SDK.ReactFlow || SDK.reactFlow || {};
  const ReactFlow = FlowSDK.ReactFlow;
  const ReactFlowProvider = FlowSDK.ReactFlowProvider;
  const Background = FlowSDK.Background;
  const Controls = FlowSDK.Controls;
  const MiniMap = FlowSDK.MiniMap;
  const Handle = FlowSDK.Handle;
  const Position = FlowSDK.Position || { Left: "left", Right: "right" };
  const MarkerType = FlowSDK.MarkerType || { ArrowClosed: "arrowclosed" };
  const addEdge = FlowSDK.addEdge;
  const applyNodeChanges = FlowSDK.applyNodeChanges;
  const applyEdgeChanges = FlowSDK.applyEdgeChanges;
  const API = "/api/plugins/workflows";
  const DEFINITIONS_API = "/api/plugins/workflows/definitions";
  const NODE_KIND_LIST = ["trigger", "pass", "switch", "agent_task", "wait", "parallel", "join", "fail"];
  const FALLBACK_IMPLEMENTED_TRIGGER_TYPES = ["manual", "schedule"];
  const FALLBACK_IMPLEMENTED_NODE_TYPES = ["pass", "switch", "agent_task", "wait", "parallel", "join", "fail"];
  const EXAMPLE_DEFINITION = [
    "id: dashboard_demo",
    "name: Dashboard Demo",
    "version: 1",
    "enabled: true",
    "triggers:",
    "  - id: manual",
    "    type: manual",
    "nodes:",
    "  start:",
    "    type: pass",
    "    output:",
    "      ok: true",
    "edges: []",
  ].join("\n");

  function api(path, options) {
    return SDK.fetchJSON(API + path, options);
  }

  function asArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function safeString(value) {
    if (value === null || value === undefined || value === "") return "—";
    return String(value);
  }

  function classSafe(value) {
    return String(value || "unknown").replace(/[^a-z0-9_-]+/gi, "-").toLowerCase();
  }

  function jsonBlock(value) {
    try {
      return JSON.stringify(value || {}, null, 2);
    } catch (_) {
      return String(value);
    }
  }

  function parseJsonObject(text) {
    try {
      const parsed = JSON.parse(text);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : null;
    } catch (_) {
      return null;
    }
  }

  function initialExecutionIdFromLocation() {
    if (typeof URLSearchParams === "undefined" || !window.location) return "";
    return new URLSearchParams(window.location.search || "").get("execution") || "";
  }

  function nodeList(spec) {
    const nodes = spec && spec.nodes ? spec.nodes : {};
    if (Array.isArray(nodes)) return nodes.map(function (node, index) {
      return Object.assign({ id: node.id || node.name || "node_" + (index + 1), specKind: "node" }, node || {});
    });
    return Object.keys(nodes).map(function (id) {
      return Object.assign({ id: id, specKind: "node" }, nodes[id] || {});
    });
  }

  function triggerList(spec) {
    return asArray(spec && spec.triggers).map(function (trigger, index) {
      const id = trigger.id || trigger.name || "trigger_" + (index + 1);
      return Object.assign({ id: id, type: "trigger", trigger_type: trigger.type, specKind: "trigger" }, trigger || {});
    });
  }

  function graphNodeList(spec) {
    return triggerList(spec).concat(nodeList(spec));
  }

  function splitPort(value) {
    const parts = safeString(value).split(".");
    return { nodeId: parts.shift() || "?", port: parts.join(".") };
  }

  function edgeList(spec) {
    return asArray(spec && spec.edges).map(function (edge, index) {
      const source = splitPort(edge.from || edge.source || edge.start || "?");
      const target = splitPort(edge.to || edge.target || edge.end || "?");
      return {
        id: edge.id || String(index + 1),
        from: source.nodeId,
        to: target.nodeId,
        label: edge.label || edge.condition || source.port || target.port || "",
        raw: edge,
      };
    });
  }

  function promptObjectiveText(prompt) {
    if (typeof prompt === "string") return prompt;
    if (Array.isArray(prompt)) return prompt.map(promptObjectiveText).filter(Boolean).join("; ");
    if (prompt && typeof prompt === "object") {
      return prompt.task || prompt.objective || prompt.title || prompt.goal || prompt.description || "";
    }
    return "";
  }

  function nodeSummaryRows(spec) {
    if (!spec || !spec.nodes) return [];
    return nodeList(spec).map(function (node) {
      const id = String(node.id || node.name || "node");
      const nextTargets = [];
      asArray(spec.edges).forEach(function (edge) {
        const source = String(edge.from || edge.from_ || edge.source || edge.start || "");
        const parts = source.split(".");
        if (parts[0] === id) {
          const target = edge.to || edge.to_ || edge.target || edge.end;
          const label = edge.label || edge.condition || parts[1] || "next";
          if (target) nextTargets.push(label === "next" ? String(target) : label + " → " + String(target));
        }
      });
      if (node.default) nextTargets.push("default → " + node.default);
      if (node.catch) nextTargets.push("catch → " + node.catch);
      const outgoing = Array.from(new Set(nextTargets.map(String))).join(", ") || "—";
      const promptObjective = promptObjectiveText(node.prompt);
      const objective = node.title || promptObjective || node.description || node.type || "—";
      return { id: id, type: node.type || "pass", profile: node.profile || "—", objective: objective, next: outgoing };
    });
  }

  function inputFieldsForSpec(spec) {
    function kindForInputValue(value) {
      function kindForLiteral(literal) {
        if (typeof literal === "boolean") return "boolean";
        if (typeof literal === "number") return "number";
        if (literal && typeof literal === "object") return "json";
        return "text";
      }
      if (Array.isArray(value)) return "json";
      if (value && typeof value === "object") {
        const type = String(value.kind || value.type || "").toLowerCase();
        if (type === "integer") return "integer";
        if (["number", "float"].indexOf(type) !== -1) return "number";
        if (["boolean", "bool"].indexOf(type) !== -1) return "boolean";
        if (["object", "array"].indexOf(type) !== -1) return "json";
        if (type === "string" || type === "text") return "text";
        if (value.default !== undefined) return kindForLiteral(value.default);
        if (value.example !== undefined) return kindForLiteral(value.example);
        return "json";
      }
      return kindForLiteral(value);
    }
    function triggerInputObject(trigger) {
      const rawInput = trigger && trigger.input;
      if (!rawInput || typeof rawInput !== "object" || Array.isArray(rawInput)) return {};
      if (rawInput.type === "object") {
        if (rawInput.properties && typeof rawInput.properties === "object" && !Array.isArray(rawInput.properties)) return rawInput.properties;
        const schemaMetadataKeys = { type: true, required: true, additionalProperties: true, description: true, title: true, $schema: true };
        if (Object.keys(rawInput).every(function (key) { return schemaMetadataKeys[key]; })) return {};
      }
      return rawInput;
    }
    const triggers = asArray(spec && spec.triggers);
    const manualTrigger = triggers.find(function (trigger) {
      if (!trigger || typeof trigger !== "object") return false;
      return String(trigger.type || trigger.trigger_type || "") === "manual" && Object.keys(triggerInputObject(trigger)).length;
    });
    const inputTrigger = manualTrigger || triggers.find(function (trigger) { return Object.keys(triggerInputObject(trigger)).length; });
    const triggerInput = inputTrigger ? triggerInputObject(inputTrigger) : {};
    const keys = Object.keys(triggerInput);
    if (keys.length) return keys.map(function (key) { return { name: key, kind: kindForInputValue(triggerInput[key]) }; });
    const found = Object.create(null);
    function addFallbackInputField(key) {
      const rawKey = String(key || "");
      const parts = rawKey.split(".");
      const name = parts[0].replace(/\[.*$/, "");
      if (!name) return;
      const kind = parts.length > 1 || rawKey.indexOf("[") !== -1 ? "json" : "text";
      if (kind === "json" || !found[name]) found[name] = kind;
    }
    const text = JSON.stringify(spec || {});
    text.replace(/\$\{\s*input\.([a-zA-Z0-9_\-.\[\]]+)\s*\}/g, function (_, key) {
      addFallbackInputField(key);
      return "";
    });
    text.replace(/\$\.input\.([a-zA-Z0-9_\-.\[\]]+)/g, function (_, key) {
      addFallbackInputField(key);
      return "";
    });
    return Object.keys(found).sort().map(function (name) { return { name: name, kind: found[name] }; });
  }

  function inputObjectForFields(fields, values) {
    const input = {};
    asArray(fields).forEach(function (field) {
      const name = field && field.name;
      if (!name) return;
      const raw = values && values[name];
      if (raw === undefined || raw === null || raw === "") return;
      if (field.kind === "number" || field.kind === "integer") {
        const errorKind = field.kind === "integer" ? "integer" : "number";
        const trimmed = String(raw).trim();
        if (!trimmed) return;
        if (field.kind === "integer" && !/^-?\d+(?:[eE][+-]?\d+)?$/.test(trimmed)) throw new Error("Invalid integer for input field " + name);
        if (field.kind !== "integer" && !/^-?(?:\d+|\d*\.\d+)(?:[eE][+-]?\d+)?$/.test(trimmed)) throw new Error("Invalid number for input field " + name);
        const numberValue = Number(trimmed);
        if (!Number.isFinite(numberValue)) throw new Error("Invalid " + errorKind + " for input field " + name);
        if (numberValue === 0 && /[1-9]/.test(trimmed.replace(/[eE].*$/, ""))) throw new Error("Invalid " + errorKind + " for input field " + name);
        if (field.kind === "integer") {
          if (!Number.isInteger(numberValue) || !Number.isSafeInteger(numberValue)) throw new Error("Invalid integer for input field " + name);
        } else {
          const fractionMatch = trimmed.match(/\.(\d+)(?:[eE][+-]?\d+)?$/);
          if (fractionMatch && /[1-9]/.test(fractionMatch[1]) && Number.isInteger(numberValue) && Math.abs(numberValue) >= Number.MAX_SAFE_INTEGER / 10) throw new Error("Invalid number for input field " + name);
          if (Number.isInteger(numberValue) && !Number.isSafeInteger(numberValue)) throw new Error("Invalid number for input field " + name);
        }
        input[name] = numberValue;
        return;
      }
      if (field.kind === "boolean") {
        if (typeof raw === "boolean") {
          input[name] = raw;
          return;
        }
        const trimmed = String(raw).trim();
        if (!trimmed) return;
        if (trimmed === "true") {
          input[name] = true;
          return;
        }
        if (trimmed === "false") {
          input[name] = false;
          return;
        }
        throw new Error("Invalid boolean for input field " + name);
      }
      if (field.kind === "json") {
        if (raw && typeof raw === "object") {
          input[name] = raw;
          return;
        }
        const trimmed = String(raw).trim();
        if (!trimmed) return;
        try {
          input[name] = JSON.parse(trimmed);
        } catch (err) {
          throw new Error("Invalid JSON for input field " + name);
        }
        return;
      }
      input[name] = raw;
    });
    return input;
  }

  function hasPromptValue(prompt) {
    if (typeof prompt === "string") return !!prompt.trim();
    if (Array.isArray(prompt)) return prompt.length > 0;
    if (prompt && typeof prompt === "object") return Object.keys(prompt).length > 0;
    return !!prompt;
  }

  function hasProfileValue(profile) {
    return typeof profile === "string" && !!profile.trim();
  }

  function checklistNodeValues(spec) {
    const nodes = spec && spec.nodes;
    if (Array.isArray(nodes)) return [];
    if (nodes && typeof nodes === "object") return Object.keys(nodes).map(function (id) { return nodes[id] && typeof nodes[id] === "object" ? nodes[id] : {}; });
    return [];
  }

  function checklistNodesShapeValid(spec) {
    const nodes = spec && spec.nodes;
    if (Array.isArray(nodes)) return false;
    if (nodes && typeof nodes === "object") return Object.keys(nodes).every(function (id) {
      const node = nodes[id];
      return node && typeof node === "object" && !Array.isArray(node);
    });
    return false;
  }

  function checklistNodeIdsValid(spec) {
    const nodes = spec && spec.nodes;
    if (!nodes || typeof nodes !== "object" || Array.isArray(nodes)) return false;
    return Object.keys(nodes).every(function (id) { return /^[a-z][a-z0-9_-]{0,63}$/.test(id); });
  }

  function checklistWorkflowIdValid(value) {
    return typeof value === "string" && /^[a-z][a-z0-9_-]{0,63}$/.test(value);
  }

  function hasStringValue(value) {
    return typeof value === "string" && !!value.trim();
  }

  function implementedTriggerTypesFromCapabilities(capabilities) {
    const values = capabilities && capabilities.triggers && capabilities.triggers.implemented;
    return Array.isArray(values) && values.length ? values : FALLBACK_IMPLEMENTED_TRIGGER_TYPES;
  }

  function implementedNodeTypesFromCapabilities(capabilities) {
    const values = capabilities && capabilities.nodes && capabilities.nodes.implemented;
    return Array.isArray(values) && values.length ? values : FALLBACK_IMPLEMENTED_NODE_TYPES;
  }

  function checklistVersionValid(version) {
    if (typeof version === "boolean" || Array.isArray(version)) return false;
    const value = Number(String(version).trim());
    return Number.isInteger(value) && value >= 1;
  }

  function checklistTriggersImplemented(spec, capabilities) {
    if (!spec || !Object.prototype.hasOwnProperty.call(spec, "triggers")) return true;
    const triggers = spec.triggers;
    if (!Array.isArray(triggers)) return false;
    const implementedTriggers = implementedTriggerTypesFromCapabilities(capabilities);
    return triggers.every(function (trigger) {
      if (!trigger || typeof trigger !== "object" || Array.isArray(trigger)) return false;
      if (typeof trigger.type !== "string") return false;
      return implementedTriggers.indexOf(trigger.type) !== -1;
    });
  }

  function checklistNodesImplemented(nodes, capabilities) {
    const implementedNodes = implementedNodeTypesFromCapabilities(capabilities);
    return !nodes.some(function (node) {
      if (!node || typeof node !== "object" || Array.isArray(node)) return true;
      if (typeof node.type !== "string") return true;
      return implementedNodes.indexOf(node.type) === -1;
    });
  }

  function checklistEdgesReferToKnownNodes(spec) {
    if (!spec || !Object.prototype.hasOwnProperty.call(spec, "edges")) return true;
    const edges = spec.edges;
    if (!Array.isArray(edges)) return false;
    if (!edges.length) return true;
    if (!checklistNodesShapeValid(spec) || !checklistNodeIdsValid(spec)) return false;
    const nodes = spec.nodes;
    function hasNode(id) {
      return Object.prototype.hasOwnProperty.call(nodes, id);
    }
    return edges.every(function (edge) {
      if (!edge || typeof edge !== "object" || Array.isArray(edge)) return false;
      const hasFrom = Object.prototype.hasOwnProperty.call(edge, "from");
      const hasFromAlias = Object.prototype.hasOwnProperty.call(edge, "from_");
      if (!hasFrom && !hasFromAlias) return false;
      if (!Object.prototype.hasOwnProperty.call(edge, "to")) return false;
      const source = String((hasFrom ? edge.from : edge.from_) || "").trim();
      const target = String(edge.to || "").trim();
      if (!source || !target || target.indexOf(".") !== -1 || !hasNode(target)) return false;
      let sourceBase = source;
      let branch = null;
      const dotIndex = source.indexOf(".");
      if (dotIndex !== -1) {
        sourceBase = source.slice(0, dotIndex);
        branch = source.slice(dotIndex + 1);
      }
      if (!sourceBase || !hasNode(sourceBase)) return false;
      const sourceType = nodes[sourceBase] && nodes[sourceBase].type;
      if (typeof sourceType !== "string") return false;
      if (branch === null) return sourceType !== "parallel";
      return !!branch && (sourceType === "switch" || sourceType === "parallel");
    });
  }

  function checklistGraphRulesValid(spec) {
    if (!spec || typeof spec !== "object" || Array.isArray(spec)) return false;

    const triggers = Object.prototype.hasOwnProperty.call(spec, "triggers") ? spec.triggers : [];
    if (!Array.isArray(triggers)) return false;
    for (let index = 0; index < triggers.length; index += 1) {
      const trigger = triggers[index];
      if (!trigger || typeof trigger !== "object" || Array.isArray(trigger)) return false;
      if (typeof trigger.type !== "string") return false;
      if (trigger.type === "schedule" && !hasStringValue(trigger.cron) && !hasStringValue(trigger.schedule) && !hasStringValue(trigger.expr)) return false;
    }

    if (!checklistNodesShapeValid(spec) || !checklistNodeIdsValid(spec)) return false;
    const nodes = spec.nodes;
    function hasNode(id) {
      return Object.prototype.hasOwnProperty.call(nodes, id);
    }

    const edges = Object.prototype.hasOwnProperty.call(spec, "edges") ? spec.edges : [];
    if (!Array.isArray(edges)) return false;
    const outgoingSources = {};
    for (let index = 0; index < edges.length; index += 1) {
      const edge = edges[index];
      if (!edge || typeof edge !== "object" || Array.isArray(edge)) return false;
      const hasFrom = Object.prototype.hasOwnProperty.call(edge, "from");
      const hasFromAlias = Object.prototype.hasOwnProperty.call(edge, "from_");
      if (!hasFrom && !hasFromAlias) return false;
      if (!Object.prototype.hasOwnProperty.call(edge, "to")) return false;
      const source = hasFrom ? edge.from : edge.from_;
      if (typeof source !== "string" || typeof edge.to !== "string") return false;
      let sourceBase = source;
      let branch = null;
      const dotIndex = source.indexOf(".");
      if (dotIndex !== -1) {
        sourceBase = source.slice(0, dotIndex);
        branch = source.slice(dotIndex + 1);
      }
      if (!hasNode(sourceBase) || !hasNode(edge.to)) return false;
      const sourceType = nodes[sourceBase].type;
      if (typeof sourceType !== "string") return false;
      if (branch === null && sourceType === "parallel") return false;
      if (branch !== null && (!branch || (sourceType !== "switch" && sourceType !== "parallel"))) return false;
      outgoingSources[sourceBase] = true;
    }

    return Object.keys(nodes).every(function (nodeId) {
      const node = nodes[nodeId];
      if (node.catch !== null && node.catch !== undefined) {
        if (typeof node.catch !== "string" || node.catch === nodeId || !hasNode(node.catch)) return false;
      }
      if (node.type === "switch") {
        if (node.default !== null && node.default !== undefined) {
          return typeof node.default === "string" && hasNode(node.default);
        }
        return !!outgoingSources[nodeId];
      }
      return true;
    });
  }

  function validationChecklist(spec, capabilities) {
    const safeNodes = checklistNodeValues(spec);
    return [
      { label: "Workflow id set", ok: checklistWorkflowIdValid(spec && spec.id) },
      { label: "Workflow name set", ok: hasStringValue(spec && spec.name) },
      { label: "Version set", ok: checklistVersionValid(spec && spec.version) },
      { label: "At least one node", ok: safeNodes.length > 0 },
      { label: "Node definitions are objects", ok: checklistNodesShapeValid(spec) },
      { label: "Node ids are valid", ok: checklistNodeIdsValid(spec) },
      { label: "No unsupported triggers (implemented today)", ok: checklistTriggersImplemented(spec, capabilities) },
      { label: "No unsupported nodes (implemented today)", ok: checklistNodesImplemented(safeNodes, capabilities) },
      { label: "Edges refer to known nodes", ok: checklistEdgesReferToKnownNodes(spec) },
      { label: "Graph rules pass", ok: checklistGraphRulesValid(spec) },
      { label: "Agent cells have profile and prompt", ok: !safeNodes.some(function (node) { return node.type === "agent_task" && (!hasProfileValue(node.profile) || !hasPromptValue(node.prompt)); }) },
    ];
  }

  function statusClass(status) {
    return "hermes-workflows-badge " + (status === false ? "is-off" : "is-on");
  }

  function eventStatus(row) {
    const payload = (row && row.payload) || {};
    const kind = String((row && row.kind) || "");
    if (payload.status) return payload.status;
    if (kind.indexOf("node_succeeded") !== -1) return "succeeded";
    if (kind.indexOf("node_failed") !== -1) return "failed";
    if (kind.indexOf("node_started") !== -1 || kind.indexOf("node_running") !== -1) return "running";
    if (kind.indexOf("execution_waiting") !== -1) return "waiting";
    return "";
  }

  function statusByNode(events) {
    const statuses = {};
    asArray(events).forEach(function (row) {
      const payload = (row && row.payload) || {};
      const status = eventStatus(row);
      asArray(payload.waiting_nodes).forEach(function (nodeId) {
        statuses[nodeId] = "waiting";
      });
      const nodeId = payload.node_id || payload.nodeId || payload.node || (payload.error && payload.error.node) || row.node_id;
      if (nodeId && status) statuses[nodeId] = status;
    });
    return statuses;
  }

  function makeWorkflowNode(kind) {
    return function WorkflowNode(props) {
      const data = (props && props.data) || {};
      const status = data.status || "idle";
      const node = data.node || {};
      return h("div", {
        className: "hermes-workflows-rf-node is-" + classSafe(kind) + " is-status-" + classSafe(status),
        onClick: function (event) {
          event.stopPropagation();
          if (data.onSelect) data.onSelect(node);
        },
      },
        Handle ? h(Handle, { type: "target", position: Position.Left }) : null,
        h("div", { className: "hermes-workflows-rf-node-title" }, safeString(node.id || data.id)),
        h("div", { className: "hermes-workflows-rf-node-type" }, kind),
        status && status !== "idle" ? h("div", { className: "hermes-workflows-rf-node-status" }, safeString(status)) : null,
        Handle ? h(Handle, { type: "source", position: Position.Right }) : null
      );
    };
  }

  const NODE_TYPES = {
    trigger: makeWorkflowNode("trigger"),
    pass: makeWorkflowNode("pass"),
    switch: makeWorkflowNode("switch"),
    agent_task: makeWorkflowNode("agent_task"),
    wait: makeWorkflowNode("wait"),
    parallel: makeWorkflowNode("parallel"),
    join: makeWorkflowNode("join"),
    fail: makeWorkflowNode("fail"),
  };

  function buildFlowNodes(spec, statuses, selectedNode, onSelect) {
    return graphNodeList(spec).map(function (node, index) {
      const id = node.id || node.name || "node_" + (index + 1);
      const kind = NODE_TYPES[node.type] ? node.type : "pass";
      return {
        id: id,
        type: kind,
        position: { x: (index % 3) * 250, y: Math.floor(index / 3) * 155 },
        className: "hermes-workflows-rf-node-shell is-status-" + classSafe(statuses[id] || "idle") + (selectedNode && selectedNode.id === id ? " is-selected" : ""),
        data: { id: id, node: node, status: statuses[id] || "idle", onSelect: onSelect },
      };
    });
  }

  function buildFlowEdges(spec) {
    return edgeList(spec).map(function (edge) {
      return {
        id: edge.id,
        source: edge.from,
        target: edge.to,
        label: edge.label,
        markerEnd: { type: MarkerType.ArrowClosed },
      };
    });
  }

  function cleanedNodeForSpec(node) {
    const cleaned = Object.assign({}, node || {});
    delete cleaned.specKind;
    delete cleaned.trigger_type;
    return cleaned;
  }

  function cloneSpec(spec) {
    return JSON.parse(JSON.stringify(spec || {}));
  }

  function specToEditorText(spec) {
    return JSON.stringify(spec || {}, null, 2);
  }

  function findSpecNode(spec, nodeId) {
    if (!spec || !nodeId) return null;
    if (Array.isArray(spec.nodes)) {
      return spec.nodes.find(function (node) { return (node.id || node.name) === nodeId; }) || null;
    }
    return spec.nodes && spec.nodes[nodeId] ? Object.assign({ id: nodeId }, spec.nodes[nodeId]) : null;
  }

  function upsertSpecNode(spec, nodeId, nextNode) {
    const next = cloneSpec(spec);
    const clean = Object.assign({}, nextNode || {});
    const nextId = clean.id || nodeId;
    delete clean.id;
    if (!nextId) return next;
    if (Array.isArray(next.nodes)) {
      const nodeWithId = Object.assign({ id: nextId }, clean);
      const sourceIndex = next.nodes.findIndex(function (node) {
        const id = node && (node.id || node.name);
        return id === nodeId;
      });
      const targetIndex = next.nodes.findIndex(function (node) {
        const id = node && (node.id || node.name);
        return id === nextId;
      });
      const index = sourceIndex >= 0 ? sourceIndex : targetIndex;
      if (index >= 0) {
        next.nodes[index] = nodeWithId;
        next.nodes = next.nodes.filter(function (node, nodeIndex) {
          const id = node && (node.id || node.name);
          return nodeIndex === index || id !== nextId;
        });
      } else next.nodes.push(nodeWithId);
      return next;
    }
    next.nodes = next.nodes || {};
    if (nodeId && nodeId !== nextId) delete next.nodes[nodeId];
    next.nodes[nextId] = clean;
    return next;
  }

  function upsertSpecEdge(spec, source, target) {
    const next = cloneSpec(spec);
    next.edges = asArray(next.edges);
    const exists = next.edges.some(function (edge) {
      return (edge.from || edge.source) === source && (edge.to || edge.target) === target;
    });
    if (!exists && source && target) next.edges.push({ from: source, to: target });
    return next;
  }

  function WorkflowsPage() {
    const useState = React.useState;
    const useEffect = React.useEffect;
    const stateDefinitions = useState([]);
    const definitions = stateDefinitions[0];
    const setDefinitions = stateDefinitions[1];
    const stateExecutions = useState([]);
    const executions = stateExecutions[0];
    const setExecutions = stateExecutions[1];
    const stateWorkflowStatus = useState(null);
    const workflowStatus = stateWorkflowStatus[0];
    const setWorkflowStatus = stateWorkflowStatus[1];
    const stateWorkflowCapabilities = useState(null);
    const workflowCapabilities = stateWorkflowCapabilities[0];
    const setWorkflowCapabilities = stateWorkflowCapabilities[1];
    const stateSelectedDefinition = useState(null);
    const selectedDefinition = stateSelectedDefinition[0];
    const setSelectedDefinition = stateSelectedDefinition[1];
    const stateSelectedExecution = useState(null);
    const selectedExecution = stateSelectedExecution[0];
    const setSelectedExecution = stateSelectedExecution[1];
    const stateSelectedNode = useState(null);
    const selectedNode = stateSelectedNode[0];
    const setSelectedNode = stateSelectedNode[1];
    const stateNodeJson = useState("");
    const nodeJson = stateNodeJson[0];
    const setNodeJson = stateNodeJson[1];
    const statePromptText = useState("");
    const promptText = statePromptText[0];
    const setPromptText = statePromptText[1];
    const stateResultContractText = useState("{}");
    const resultContractText = stateResultContractText[0];
    const setResultContractText = stateResultContractText[1];
    const stateAgentProfile = useState("");
    const agentProfile = stateAgentProfile[0];
    const setAgentProfile = stateAgentProfile[1];
    const stateAgentTitle = useState("");
    const agentTitle = stateAgentTitle[0];
    const setAgentTitle = stateAgentTitle[1];
    const stateAdvancedJsonOpen = useState(false);
    const advancedJsonOpen = stateAdvancedJsonOpen[0];
    const setAdvancedJsonOpen = stateAdvancedJsonOpen[1];
    const statePromptAssistantOpen = useState(false);
    const promptAssistantOpen = statePromptAssistantOpen[0];
    const setPromptAssistantOpen = statePromptAssistantOpen[1];
    const statePromptAssistantGoal = useState("");
    const promptAssistantGoal = statePromptAssistantGoal[0];
    const setPromptAssistantGoal = statePromptAssistantGoal[1];
    const statePromptAssistantObjective = useState("");
    const promptAssistantObjective = statePromptAssistantObjective[0];
    const setPromptAssistantObjective = statePromptAssistantObjective[1];
    const statePromptAssistantContext = useState("${ input }\n${ node.previous.output }");
    const promptAssistantContext = statePromptAssistantContext[0];
    const setPromptAssistantContext = statePromptAssistantContext[1];
    const statePromptAssistantOutput = useState('{"summary":"string","status":"string"}');
    const promptAssistantOutput = statePromptAssistantOutput[0];
    const setPromptAssistantOutput = statePromptAssistantOutput[1];
    const statePromptAssistantConstraints = useState("Return JSON only");
    const promptAssistantConstraints = statePromptAssistantConstraints[0];
    const setPromptAssistantConstraints = statePromptAssistantConstraints[1];
    const stateNodeMessage = useState("");
    const nodeMessage = stateNodeMessage[0];
    const setNodeMessage = stateNodeMessage[1];
    const stateFlowNodes = useState([]);
    const flowNodes = stateFlowNodes[0];
    const setFlowNodes = stateFlowNodes[1];
    const stateFlowEdges = useState([]);
    const flowEdges = stateFlowEdges[0];
    const setFlowEdges = stateFlowEdges[1];
    const stateEditorText = useState(EXAMPLE_DEFINITION);
    const editorText = stateEditorText[0];
    const setEditorText = stateEditorText[1];
    const stateDraftSpec = useState(null);
    const draftSpec = stateDraftSpec[0];
    const setDraftSpec = stateDraftSpec[1];
    const stateGoalText = useState("");
    const goalText = stateGoalText[0];
    const setGoalText = stateGoalText[1];
    const stateDraftResult = useState(null);
    const draftResult = stateDraftResult[0];
    const setDraftResult = stateDraftResult[1];
    const stateDrafting = useState(false);
    const drafting = stateDrafting[0];
    const setDrafting = stateDrafting[1];
    const stateRefineText = useState("");
    const refineText = stateRefineText[0];
    const setRefineText = stateRefineText[1];
    const stateRefining = useState(false);
    const refining = stateRefining[0];
    const setRefining = stateRefining[1];
    const stateShowAdvancedYaml = useState(false);
    const showAdvancedYaml = stateShowAdvancedYaml[0];
    const setShowAdvancedYaml = stateShowAdvancedYaml[1];
    const stateRunWorkflowId = useState("");
    const runWorkflowId = stateRunWorkflowId[0];
    const setRunWorkflowId = stateRunWorkflowId[1];
    const stateRunInputText = useState("{}");
    const runInputText = stateRunInputText[0];
    const setRunInputText = stateRunInputText[1];
    const stateInputFieldValues = useState({});
    const inputFieldValues = stateInputFieldValues[0];
    const setInputFieldValues = stateInputFieldValues[1];
    const stateShowAdvancedInputJson = useState(false);
    const showAdvancedInputJson = stateShowAdvancedInputJson[0];
    const setShowAdvancedInputJson = stateShowAdvancedInputJson[1];
    const stateEvents = useState([]);
    const events = stateEvents[0];
    const setEvents = stateEvents[1];
    const stateStatus = useState("");
    const status = stateStatus[0];
    const setStatus = stateStatus[1];
    const stateError = useState("");
    const error = stateError[0];
    const setError = stateError[1];
    const stateLoading = useState(false);
    const loading = stateLoading[0];
    const setLoading = stateLoading[1];
    const stateValidating = useState(false);
    const validating = stateValidating[0];
    const setValidating = stateValidating[1];
    const stateDeploying = useState(false);
    const deploying = stateDeploying[0];
    const setDeploying = stateDeploying[1];
    const stateRunning = useState(false);
    const running = stateRunning[0];
    const setRunning = stateRunning[1];
    const initialExecutionId = initialExecutionIdFromLocation();

    function fail(err) {
      setError(err && err.message ? err.message : String(err));
    }

    function updateEditorText(text) {
      setEditorText(text);
      if (!parseJsonObject(text)) setDraftSpec(null);
    }

    function activeSpec() {
      return parseJsonObject(editorText) || draftSpec || null;
    }

    function checklistSpec() {
      return parseJsonObject(editorText) || draftSpec || null;
    }

    function workflowIdForSpec(spec) {
      return spec && String(spec.workflow_id || spec.id || "");
    }

    function runInputSpec() {
      const selectedId = selectedDefinition && String(selectedDefinition.workflow_id || selectedDefinition.id || "");
      if (selectedDefinition && selectedId === String(runWorkflowId || "")) return selectedDefinition.spec || null;
      const spec = activeSpec();
      if (spec && workflowIdForSpec(spec) === String(runWorkflowId || "")) return spec;
      return null;
    }

    function selectNodeForInspector(node) {
      setSelectedNode(node);
      setNodeJson(jsonBlock(node));
      setNodeMessage("");
      setAdvancedJsonOpen(false);
      setPromptAssistantOpen(false);
      const rawPrompt = node ? node.prompt : undefined;
      const assistantOutput = node && node.result_contract ? jsonBlock(node.result_contract) : '{"summary":"string","status":"string"}';
      setAgentProfile(node && node.profile ? String(node.profile) : "");
      setAgentTitle(node && node.title ? String(node.title) : "");
      setPromptText(rawPrompt === null || rawPrompt === undefined ? "" : (typeof rawPrompt === "string" ? rawPrompt : jsonBlock(rawPrompt)));
      setResultContractText(jsonBlock((node && node.result_contract) || {}));
      setPromptAssistantObjective(node && node.title ? String(node.title) : "");
      setPromptAssistantContext("${ input }\n${ node.previous.output }");
      setPromptAssistantOutput(assistantOutput);
      setPromptAssistantConstraints("Return JSON only");
    }

    function loadDefinition(workflowId) {
      if (!workflowId) {
        setSelectedDefinition(null);
        setDraftSpec(null);
        setSelectedNode(null);
        return Promise.resolve(null);
      }
      return api("/definitions/" + encodeURIComponent(workflowId)).then(function (res) {
        const definition = res.definition || null;
        setSelectedDefinition(definition);
        setDraftSpec(definition && definition.spec ? definition.spec : null);
        setInputFieldValues({});
        setShowAdvancedInputJson(false);
        setRunInputText("{}");
        setDraftResult(null);
        setRefineText("");
        if (definition && definition.spec) updateEditorText(specToEditorText(definition.spec));
        setSelectedNode(null);
        if (definition) setRunWorkflowId(definition.workflow_id || definition.id || workflowId);
        return definition;
      });
    }

    function loadEvents(executionId) {
      if (!executionId) {
        setEvents([]);
        return Promise.resolve([]);
      }
      return api("/executions/" + encodeURIComponent(executionId) + "/events").then(function (res) {
        const rows = asArray(res.events);
        setEvents(rows);
        return rows;
      });
    }

    function loadExecution(executionId) {
      if (!executionId) {
        setSelectedExecution(null);
        setEvents([]);
        return Promise.resolve(null);
      }
      return api("/executions/" + encodeURIComponent(executionId)).then(function (res) {
        const execution = res.execution || null;
        setSelectedExecution(execution);
        return loadEvents(executionId).then(function () { return execution; });
      });
    }

    function loadDefinitions(preferId) {
      return SDK.fetchJSON(DEFINITIONS_API).then(function (res) {
        const rows = asArray(res.definitions);
        const currentId = selectedDefinition && (selectedDefinition.workflow_id || selectedDefinition.id);
        const nextId = preferId || currentId || runWorkflowId || (rows[0] && (rows[0].workflow_id || rows[0].id)) || "";
        setDefinitions(rows);
        if (nextId) return loadDefinition(nextId);
        setRunWorkflowId("");
        setSelectedDefinition(null);
        setDraftSpec(null);
        setSelectedNode(null);
        return null;
      });
    }

    function loadExecutions(preferId) {
      return api("/executions").then(function (res) {
        const rows = asArray(res.executions);
        const currentId = selectedExecution && selectedExecution.execution_id;
        const nextId = preferId || currentId || (rows[0] && rows[0].execution_id) || "";
        setExecutions(rows);
        if (nextId) return loadExecution(nextId);
        setSelectedExecution(null);
        setEvents([]);
        return null;
      });
    }

    function loadWorkflowStatus() {
      return api("/status").then(setWorkflowStatus).catch(function () {
        setWorkflowStatus(null);
      });
    }

    function loadWorkflowCapabilities() {
      return api("/capabilities").then(setWorkflowCapabilities).catch(function () {
        setWorkflowCapabilities(null);
      });
    }

    function refresh(preferExecutionId) {
      setLoading(true);
      setError("");
      loadWorkflowStatus();
      loadWorkflowCapabilities();
      return Promise.all([loadDefinitions(), loadExecutions(preferExecutionId)])
        .catch(fail)
        .finally(function () { setLoading(false); });
    }

    useEffect(function () {
      refresh(initialExecutionId);
    }, []);

    useEffect(function () {
      const spec = activeSpec();
      const statuses = statusByNode(events);
      setFlowNodes(spec ? buildFlowNodes(spec, statuses, selectedNode, selectNodeForInspector) : []);
      setFlowEdges(spec ? buildFlowEdges(spec) : []);
    }, [draftSpec, editorText, events, selectedNode]);

    function validateDefinition() {
      setValidating(true);
      setError("");
      api("/definitions/validate", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: editorText,
      }).then(function (res) {
        const definition = res.definition || {};
        setStatus("Validated " + safeString(definition.workflow_id || definition.id));
        setDraftSpec(definition.spec || null);
        if (definition.spec) setSelectedDefinition(definition);
      }).catch(fail).finally(function () { setValidating(false); });
    }

    function deployDefinition() {
      setDeploying(true);
      setError("");
      api("/definitions/deploy", {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: editorText,
      }).then(function (res) {
        const definition = res.definition || {};
        const id = definition.workflow_id || definition.id || "";
        setDraftSpec(definition.spec || null);
        setStatus("Deployed " + safeString(id));
        return loadDefinitions(id);
      }).catch(fail).finally(function () { setDeploying(false); });
    }

    function runWorkflow(event) {
      event.preventDefault();
      const workflowId = (runWorkflowId || "").trim();
      if (!workflowId) {
        setError("Choose a workflow before running it.");
        return;
      }
      let input = {};
      try {
        if (showAdvancedInputJson) {
          input = JSON.parse(runInputText || "{}");
        } else {
          input = inputObjectForFields(inputFieldsForSpec(runInputSpec()), inputFieldValues);
        }
      } catch (err) {
        fail(err);
        return;
      }
      setRunning(true);
      setError("");
      api("/definitions/" + encodeURIComponent(workflowId) + "/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: input }),
      }).then(function (res) {
        const execution = res.execution || {};
        const executionId = execution.execution_id;
        setStatus("Started execution " + safeString(executionId));
        return loadExecutions(executionId);
      }).catch(fail).finally(function () { setRunning(false); });
    }

    function draftFromGoal(event) {
      event.preventDefault();
      const goal = (goalText || "").trim();
      setStatus("");
      setDraftResult(null);
      if (!goal) {
        setError("Describe what you want the workflow to automate.");
        return;
      }
      setDrafting(true);
      setError("");
      api("/definitions/draft", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal: goal }),
      }).then(function (res) {
        const draft = res.draft || res;
        setDraftResult(draft);
        if (draft.spec) {
          setSelectedDefinition(null);
          setSelectedNode(null);
          setNodeJson("");
          setNodeMessage("");
          setDraftSpec(draft.spec);
          setInputFieldValues({});
          setShowAdvancedInputJson(false);
          setRunInputText("{}");
          updateEditorText(specToEditorText(draft.spec));
        }
        setStatus("Drafted workflow from goal. Review the plan before deploy.");
      }).catch(fail).finally(function () { setDrafting(false); });
    }

    function refineWorkflow(event) {
      if (event) event.preventDefault();
      const instruction = (refineText || "").trim();
      const spec = activeSpec();
      setStatus("");
      setDraftResult(null);
      if (!instruction || !spec) {
        setError("Select or draft a workflow, then describe the refinement.");
        return;
      }
      setRefining(true);
      setError("");
      api("/definitions/refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec: spec, instruction: instruction }),
      }).then(function (res) {
        const draft = (res && res.draft) || res || {};
        if (!draft.spec) throw new Error("Refine response did not include a workflow spec.");
        setDraftResult(draft);
        setSelectedDefinition(null);
        setSelectedNode(null);
        setNodeJson("");
        setNodeMessage("");
        setDraftSpec(draft.spec);
        setInputFieldValues({});
        setShowAdvancedInputJson(false);
        setRunInputText("{}");
        updateEditorText(specToEditorText(draft.spec));
        setRefineText("");
        setStatus("Refined workflow draft.");
      }).catch(fail).finally(function () { setRefining(false); });
    }

    function importDefinitionFile(event) {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function () {
        updateEditorText(String(reader.result || ""));
        setStatus("Import YAML loaded " + safeString(file.name));
        setDraftResult(null);
        setInputFieldValues({});
        setShowAdvancedInputJson(false);
        setRunInputText("{}");
        setSelectedDefinition(null);
        setNodeJson("");
        setNodeMessage("");
        setSelectedNode(null);
      };
      reader.onerror = function () { setError("Could not import workflow file."); };
      reader.readAsText(file);
      event.target.value = "";
    }

    function exportYAML() {
      const spec = activeSpec();
      const base = (spec && (spec.id || spec.workflow_id || spec.name)) || runWorkflowId || "workflow";
      const blob = new Blob([editorText], { type: "text/yaml;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = classSafe(base) + ".yaml";
      anchor.click();
      setTimeout(function () { URL.revokeObjectURL(url); }, 0);
      setStatus("Export YAML downloaded " + anchor.download);
    }

    function copyYAML() {
      if (!navigator.clipboard) {
        setStatus("Clipboard API unavailable; use Export YAML instead.");
        return;
      }
      navigator.clipboard.writeText(editorText).then(function () {
        setStatus("Export YAML copied to clipboard.");
      }).catch(fail);
    }

    function useJsonDraft() {
      const spec = parseJsonObject(editorText) || draftSpec;
      if (!spec) {
        setNodeMessage("Validate the YAML draft before converting it to JSON; no stale workflow was used.");
        return;
      }
      updateEditorText(JSON.stringify(spec, null, 2));
      setNodeMessage("Converted current workflow draft to JSON; node edits can now be applied.");
    }

    function applyNodeJson() {
      if (!selectedNode) return;
      let nextNode;
      try {
        nextNode = JSON.parse(nodeJson);
      } catch (err) {
        setNodeMessage("Invalid node JSON: " + (err && err.message ? err.message : String(err)));
        return;
      }

      const spec = parseJsonObject(editorText);
      if (!spec) {
        setNodeMessage("Current definition text is YAML. Convert to JSON before applying node JSON; YAML was not changed.");
        return;
      }

      const clean = cleanedNodeForSpec(nextNode);
      const nextId = clean.id || selectedNode.id;
      if (selectedNode.specKind === "trigger") {
        spec.triggers = asArray(spec.triggers).map(function (trigger) {
          const triggerId = trigger.id || trigger.name;
          return triggerId === selectedNode.id ? clean : trigger;
        });
      } else if (Array.isArray(spec.nodes)) {
        spec.nodes = spec.nodes.map(function (node) {
          const nodeId = node.id || node.name;
          return nodeId === selectedNode.id ? clean : node;
        });
      } else {
        spec.nodes = spec.nodes || {};
        if (nextId !== selectedNode.id) delete spec.nodes[selectedNode.id];
        const nodeValue = Object.assign({}, clean);
        delete nodeValue.id;
        spec.nodes[nextId] = nodeValue;
      }

      updateEditorText(JSON.stringify(spec, null, 2));
      setSelectedDefinition(Object.assign({}, selectedDefinition || {}, { spec: spec }));
      setSelectedNode(Object.assign({}, nextNode, { id: nextId, specKind: selectedNode.specKind }));
      setNodeMessage("Applied node JSON to editor draft.");
    }

    function applyAgentCellForm() {
      if (!selectedNode) return;
      const spec = activeSpec();
      if (!spec) {
        setNodeMessage("Validate the YAML draft before applying cell edits; no stale workflow was used.");
        return;
      }
      const nextNode = Object.assign({}, selectedNode, {
        type: "agent_task",
        profile: agentProfile.trim(),
        title: agentTitle.trim() || selectedNode.id,
        prompt: promptText,
      });
      const contractText = resultContractText.trim();
      if (contractText) {
        const contract = parseJsonObject(contractText);
        if (!contract) {
          setNodeMessage("Result contract JSON must be a JSON object.");
          return;
        }
        nextNode.result_contract = contract;
      } else {
        delete nextNode.result_contract;
      }
      const nextSpec = upsertSpecNode(spec, selectedNode.id, cleanedNodeForSpec(nextNode));
      updateEditorText(specToEditorText(nextSpec));
      setDraftSpec(nextSpec);
      setSelectedDefinition(Object.assign({}, selectedDefinition || {}, { spec: nextSpec }));
      setSelectedNode(nextNode);
      setNodeJson(jsonBlock(nextNode));
      setNodeMessage("Applied agent cell prompt to workflow draft.");
    }

    function draftPromptWithAssistant() {
      const outputText = promptAssistantOutput.trim();
      const expectedOutput = outputText ? parseJsonObject(outputText) : {};
      if (outputText && !expectedOutput) {
        setNodeMessage("Expected output contract JSON must be a JSON object.");
        return;
      }
      api("/prompt-assistant/draft", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workflow_goal: promptAssistantGoal,
          node_id: selectedNode && selectedNode.id,
          profile: agentProfile,
          cell_objective: promptAssistantObjective,
          available_context: promptAssistantContext.split(/\n+/).map(function (line) { return line.trim(); }).filter(Boolean),
          expected_output: expectedOutput,
          constraints: promptAssistantConstraints.split(/\n+/).map(function (line) { return line.trim(); }).filter(Boolean),
        }),
      }).then(function (res) {
        setPromptText(res.prompt_text || "");
        setResultContractText(jsonBlock(res.result_contract || expectedOutput));
        setNodeMessage("Prompt assistant drafted a cell prompt. Review it, then Apply cell prompt.");
      }).catch(fail);
    }

    function renderGoalBuilder() {
      const templates = [
        ["Code change + review", "Change code in a repository, run the tests, review the diff, and report whether it is ready."],
        ["Research triage", "Collect sources for a research question, summarize findings, flag gaps, and produce a recommendation."],
        ["Daily briefing", "Every morning, gather updates from configured sources and produce a concise briefing with action items."],
        ["Human approval loop", "Prepare a proposed action, wait for human approval, then continue only if it is approved."],
      ];
      return h(Card, { className: "hermes-workflows-panel hermes-workflows-goal" },
        h("h2", null, "What do you want to automate?"),
        h("p", { className: "hermes-workflows-muted" }, "Describe the outcome in plain language. Hermes drafts the graph, cells, inputs, and output contracts for you to review."),
        h("form", { className: "hermes-workflows-stack", onSubmit: draftFromGoal },
          h("textarea", {
            className: "hermes-workflows-goal-input",
            "aria-label": "Describe workflow goal",
            value: goalText,
            onChange: function (event) { setGoalText(event.target.value); },
            placeholder: "Example: review code changes, run tests, ask for approval, then deploy if approved.",
          }),
          h("div", { className: "hermes-workflows-row" },
            h("button", { type: "submit", disabled: drafting, className: "hermes-workflows-primary" }, drafting ? "Drafting…" : "Describe workflow"),
            h("button", { type: "button", onClick: function () { setShowAdvancedYaml(!showAdvancedYaml); } }, showAdvancedYaml ? "Hide Advanced YAML" : "Advanced YAML")
          )
        ),
        h("p", { className: "hermes-workflows-muted" }, "Use Kanban for one-off work queues; use workflows when the same automation should run repeatedly."),
        h("div", { className: "hermes-workflows-template-grid" }, templates.map(function (template) {
          return h("button", {
            key: template[0],
            type: "button",
            className: "hermes-workflows-template-card",
            onClick: function () { setGoalText(template[1]); },
          }, template[0]);
        }))
      );
    }

    function renderDraftReview() {
      const spec = activeSpec();
      if (!draftResult && !spec) return null;
      const rows = nodeSummaryRows(spec);
      const hasDraftMetadata = !!draftResult;

      function renderNotes(title, items) {
        const values = asArray(items).filter(Boolean);
        return h("div", { className: "hermes-workflows-draft-section" },
          h("h3", null, title),
          values.length ? h("ul", null, values.map(function (item, index) {
            return h("li", { key: title + index }, safeString(item));
          })) : h("p", { className: "hermes-workflows-muted" }, "None reported.")
        );
      }

      return h(Card, { className: "hermes-workflows-panel hermes-workflows-draft-review" },
        h("div", null,
          h("h2", null, "Draft review"),
          h("p", { className: "hermes-workflows-muted" }, "Review the workflow cells in plain language before using Advanced YAML or deploying.")
        ),
        draftResult && draftResult.summary ? h("p", { className: "hermes-workflows-draft-summary" }, safeString(draftResult.summary)) : null,
        rows.length ? h("div", { className: "hermes-workflows-draft-table-wrap" },
          h("table", { className: "hermes-workflows-draft-table" },
            h("thead", null,
              h("tr", null,
                h("th", null, "Cell"),
                h("th", null, "Type"),
                h("th", null, "Profile"),
                h("th", null, "Objective"),
                h("th", null, "Next")
              )
            ),
            h("tbody", null, rows.map(function (row) {
              return h("tr", { key: row.id },
                h("td", null, safeString(row.id)),
                h("td", null, safeString(row.type)),
                h("td", null, safeString(row.profile)),
                h("td", null, safeString(row.objective)),
                h("td", null, safeString(row.next))
              );
            }))
          )
        ) : h("p", { className: "hermes-workflows-muted" }, "No workflow cells to review yet."),
        hasDraftMetadata ? h("div", { className: "hermes-workflows-draft-sections" },
          renderNotes("Questions", draftResult && draftResult.questions),
          renderNotes("Assumptions", draftResult && draftResult.assumptions),
          renderNotes("Warnings", draftResult && draftResult.warnings),
          renderNotes("Unsupported requests", draftResult && draftResult.unsupported_requests)
        ) : h("p", { className: "hermes-workflows-muted" }, "No assistant draft metadata available."),
        h("form", { className: "hermes-workflows-stack hermes-workflows-refine-form", onSubmit: refineWorkflow },
          h("label", null,
            h("span", { className: "hermes-workflows-muted" }, "Refine workflow"),
            h("textarea", {
              className: "hermes-workflows-refine-input",
              value: refineText,
              onChange: function (event) { setRefineText(event.target.value); },
              placeholder: "Example: add a human approval step before deployment.",
              "aria-label": "Refine workflow",
            })
          ),
          h("button", { type: "submit", disabled: refining, className: "hermes-workflows-primary" }, refining ? "Refining…" : "Refine workflow")
        )
      );
    }

    function renderValidationChecklist() {
      const spec = checklistSpec();
      if (!spec) {
        return h(Card, { className: "hermes-workflows-panel hermes-workflows-validation-checklist" },
          h("h2", null, "Validation checklist"),
          h("p", { className: "hermes-workflows-muted" }, "Validate Advanced YAML to update the checklist.")
        );
      }
      const implementedTriggers = implementedTriggerTypesFromCapabilities(workflowCapabilities);
      const implementedNodes = implementedNodeTypesFromCapabilities(workflowCapabilities);
      const items = validationChecklist(spec, workflowCapabilities);
      return h(Card, { className: "hermes-workflows-panel hermes-workflows-validation-checklist" },
        h("h2", null, "Validation checklist"),
        h("p", { className: "hermes-workflows-muted" }, "This checklist checks implemented dashboard/dispatcher readiness, not every declared WorkflowSpec primitive."),
        h("ul", { className: "hermes-workflows-checklist" }, items.map(function (item) {
          return h("li", { key: item.label, className: "hermes-workflows-checklist-item " + (item.ok ? "is-ok" : "is-fail") },
            h("span", { className: "hermes-workflows-checklist-mark", "aria-hidden": "true" }, item.ok ? "✓" : "!"),
            h("span", null, item.label)
          );
        })),
        h("p", { className: "hermes-workflows-muted" }, "Implemented triggers today: " + implementedTriggers.join(", ") + "."),
        h("p", { className: "hermes-workflows-muted" }, "Implemented node types today: " + implementedNodes.join(", ") + ".")
      );
    }

    function renderDispatcherReadiness() {
      const dispatcher = workflowStatus && workflowStatus.dispatcher ? workflowStatus.dispatcher : {};
      const statusKnown = dispatcher.status_available !== false && typeof dispatcher.dispatch_in_gateway === "boolean";
      const ready = statusKnown && dispatcher.dispatch_in_gateway === true;
      const tick = dispatcher.tick_interval_seconds;
      const className = "hermes-workflows-panel hermes-workflows-dispatcher-readiness " + (statusKnown ? (ready ? "is-ready" : "is-warning") : "is-unknown");
      if (!statusKnown) {
        return h(Card, { className: className },
          h("h2", null, "Dispatcher readiness"),
          h("p", null, "Dispatcher readiness unavailable."),
          h("p", { className: "hermes-workflows-muted" }, "Status endpoint did not report dispatcher readiness.")
        );
      }
      const warning = dispatcher.warning || "Set workflow.dispatch_in_gateway: true to let the gateway advance runs; fallback: hermes workflow tick.";
      return h(Card, { className: className },
        h("h2", null, "Dispatcher readiness"),
        ready ? h("p", null, "Ready: gateway dispatcher is advancing workflow runs.") : h("p", null, safeString(warning)),
        ready ? h("p", { className: "hermes-workflows-muted" }, "Tick interval: " + safeString(tick) + "s") : h("p", { className: "hermes-workflows-muted" }, "Enable workflow.dispatch_in_gateway or run hermes workflow tick manually.")
      );
    }

    function renderAdvancedYaml() {
      if (!showAdvancedYaml) return null;
      return h(Card, { className: "hermes-workflows-panel" },
        h("h2", null, "Advanced YAML"),
        h("textarea", {
          className: "hermes-workflows-editor",
          value: editorText,
          onChange: function (event) {
            setDraftResult(null);
            setInputFieldValues({});
            setShowAdvancedInputJson(false);
            setRunInputText("{}");
            updateEditorText(event.target.value);
          },
        }),
        h("div", { className: "hermes-workflows-row" },
          h("button", { type: "button", disabled: validating, onClick: validateDefinition }, validating ? "Validating…" : "Validate"),
          h("button", { type: "button", disabled: deploying, onClick: deployDefinition, className: "hermes-workflows-primary" }, deploying ? "Deploying…" : "Deploy"),
          h("label", { className: "hermes-workflows-file-button" },
            "Import YAML",
            h("input", { type: "file", accept: ".yaml,.yml,.json,application/yaml,application/x-yaml,application/json", onChange: importDefinitionFile })
          ),
          h("button", { type: "button", onClick: exportYAML }, "Export YAML"),
          h("button", { type: "button", onClick: copyYAML }, "Copy YAML")
        )
      );
    }

    function renderDefinitionList() {
      return h("div", { className: "hermes-workflows-list" },
        definitions.length ? definitions.map(function (definition) {
          const id = definition.workflow_id || definition.id;
          const selectedId = selectedDefinition && (selectedDefinition.workflow_id || selectedDefinition.id);
          return h("button", {
            key: id + ":" + safeString(definition.version),
            type: "button",
            className: "hermes-workflows-item" + (id === selectedId ? " is-selected" : ""),
            onClick: function () {
              setError("");
              loadDefinition(id).catch(fail);
            },
          },
            h("div", { className: "hermes-workflows-item-title" },
              h("span", null, safeString(definition.name || id)),
              h("span", { className: statusClass(definition.enabled) }, definition.enabled ? "enabled" : "disabled")
            ),
            h("div", { className: "hermes-workflows-meta" }, safeString(id) + " · v" + safeString(definition.version))
          );
        }) : h("p", { className: "hermes-workflows-muted" }, "No workflow definitions deployed yet.")
      );
    }

    function renderRunInputForm() {
      const spec = runInputSpec();
      const fields = inputFieldsForSpec(spec);
      return h(Card, { className: "hermes-workflows-panel hermes-workflows-run-form" },
        h("h2", null, "Run test"),
        h("form", { className: "hermes-workflows-stack", onSubmit: runWorkflow },
          h("label", null,
            h("span", { className: "hermes-workflows-muted" }, "Workflow id"),
            definitions.length ? h("select", {
              value: runWorkflowId,
              onChange: function (event) {
                setRunWorkflowId(event.target.value);
                loadDefinition(event.target.value).catch(fail);
              },
            }, definitions.map(function (definition) {
              const id = definition.workflow_id || definition.id;
              return h("option", { key: id, value: id }, id);
            })) : h("input", {
              value: runWorkflowId,
              onChange: function (event) { setRunWorkflowId(event.target.value); },
              placeholder: "workflow_id",
            })
          ),
          fields.length && !showAdvancedInputJson ? fields.map(function (field) {
            const value = inputFieldValues[field.name] === undefined ? "" : inputFieldValues[field.name];
            const onChange = function (event) {
              const next = Object.assign({}, inputFieldValues);
              next[field.name] = event.target.value;
              setInputFieldValues(next);
            };
            const control = field.kind === "boolean" ? h("select", {
              value: value,
              onChange: onChange,
            },
              h("option", { value: "" }, ""),
              h("option", { value: "true" }, "true"),
              h("option", { value: "false" }, "false")
            ) : field.kind === "json" ? h("textarea", {
              className: "hermes-workflows-run-input",
              placeholder: "{} or []",
              value: value,
              onChange: onChange,
            }) : h("input", {
              type: field.kind === "number" || field.kind === "integer" ? "number" : "text",
              step: field.kind === "number" ? "any" : field.kind === "integer" ? "1" : undefined,
              value: value,
              onChange: onChange,
            });
            return h("label", { key: field.name },
              h("span", { className: "hermes-workflows-muted" }, field.name),
              control
            );
          }) : null,
          !fields.length && !showAdvancedInputJson ? h("p", { className: "hermes-workflows-muted" }, "No input fields detected; this test run will send an empty input object.") : null,
          h("button", {
            type: "button",
            onClick: function () { setShowAdvancedInputJson(!showAdvancedInputJson); },
          }, showAdvancedInputJson ? "Hide Advanced input JSON" : "Advanced input JSON"),
          showAdvancedInputJson ? h("label", null,
            h("span", { className: "hermes-workflows-muted" }, "Input JSON"),
            h("textarea", {
              className: "hermes-workflows-run-input",
              value: runInputText,
              onChange: function (event) { setRunInputText(event.target.value); },
            })
          ) : null,
          h("button", { type: "submit", disabled: running, className: "hermes-workflows-primary" }, running ? "Running…" : "Run workflow")
        )
      );
    }

    function renderExecutions() {
      return h("div", { className: "hermes-workflows-executions" },
        executions.length ? executions.map(function (execution) {
          const id = execution.execution_id;
          return h("button", {
            key: id,
            type: "button",
            className: "hermes-workflows-item" + (selectedExecution && selectedExecution.execution_id === id ? " is-selected" : ""),
            onClick: function () {
              setError("");
              loadExecution(id).catch(fail);
            },
          },
            h("div", { className: "hermes-workflows-item-title" },
              h("span", null, safeString(id)),
              h("span", { className: "hermes-workflows-badge" }, safeString(execution.status))
            ),
            h("div", { className: "hermes-workflows-meta" },
              safeString(execution.workflow_id) + " · " + safeString(execution.updated_at || execution.created_at)
            )
          );
        }) : h("p", { className: "hermes-workflows-muted" }, "No executions yet.")
      );
    }

    function renderTimeline() {
      return h("div", { className: "hermes-workflows-timeline" },
        selectedExecution ? h("div", { className: "hermes-workflows-event" },
          h("div", { className: "hermes-workflows-item-title" },
            h("strong", null, safeString(selectedExecution.execution_id)),
            h("span", { className: "hermes-workflows-badge" }, safeString(selectedExecution.status))
          ),
          h("div", { className: "hermes-workflows-meta" },
            safeString(selectedExecution.workflow_id) + " · created " + safeString(selectedExecution.created_at)
          ),
          h("pre", { className: "hermes-workflows-pre" }, jsonBlock(selectedExecution.input))
        ) : h("p", { className: "hermes-workflows-muted" }, "Select an execution to inspect it."),
        events.length ? events.map(function (row) {
          return h("div", { key: row.id, className: "hermes-workflows-event" },
            h("div", { className: "hermes-workflows-item-title" },
              h("span", { className: "hermes-workflows-event-kind" }, safeString(row.kind)),
              h("span", { className: "hermes-workflows-meta" }, safeString(row.created_at))
            ),
            row.node_run_id ? h("div", { className: "hermes-workflows-meta" }, "node run " + row.node_run_id) : null,
            h("pre", { className: "hermes-workflows-pre" }, jsonBlock(row.payload))
          );
        }) : h("p", { className: "hermes-workflows-muted" }, "No events recorded for this execution.")
      );
    }

    function renderSimpleGraph(spec) {
      const nodes = graphNodeList(spec);
      const edges = edgeList(spec);
      return h("div", { className: "hermes-workflows-graph-fallback" },
        nodes.length ? h("div", { className: "hermes-workflows-node-grid" }, nodes.map(function (node) {
          const id = node.id || node.name || "node";
          return h("div", {
            key: id,
            className: "hermes-workflows-node-card",
            onClick: function () { selectNodeForInspector(node); },
          },
            h("h3", null, safeString(id)),
            h("div", { className: "hermes-workflows-node-type" }, safeString(node.type)),
            h("pre", { className: "hermes-workflows-pre" }, jsonBlock(node))
          );
        })) : h("p", { className: "hermes-workflows-muted" }, "No nodes to render."),
        h("div", { className: "hermes-workflows-stack" },
          h("h3", null, "Edges"),
          edges.length ? edges.map(function (edge) {
            const text = safeString(edge.from) + " → " + safeString(edge.to) + (edge.label ? " · " + edge.label : "");
            return h("div", { key: edge.id, className: "hermes-workflows-edge-card" }, text);
          }) : h("p", { className: "hermes-workflows-muted" }, "No edges defined.")
        )
      );
    }

    function renderAdvancedNodeJson(spec) {
      return h("div", { className: "hermes-workflows-stack" },
        h("h3", null, "Advanced JSON"),
        h("textarea", {
          className: "hermes-workflows-node-json",
          value: nodeJson,
          onChange: function (event) { setNodeJson(event.target.value); },
        }),
        h("div", { className: "hermes-workflows-row" },
          h("button", { type: "button", onClick: applyNodeJson }, "Apply node JSON"),
          h("button", { type: "button", onClick: useJsonDraft, disabled: !spec }, "Use JSON draft")
        )
      );
    }

    function renderPromptAssistant() {
      if (!promptAssistantOpen) return null;
      return h("div", { className: "hermes-workflows-assistant" },
        h("h4", null, "Prompt assistant"),
        h("label", null,
          h("span", { className: "hermes-workflows-muted" }, "Workflow goal"),
          h("textarea", { value: promptAssistantGoal, onChange: function (event) { setPromptAssistantGoal(event.target.value); } })
        ),
        h("label", null,
          h("span", { className: "hermes-workflows-muted" }, "Cell objective"),
          h("textarea", { value: promptAssistantObjective, onChange: function (event) { setPromptAssistantObjective(event.target.value); } })
        ),
        h("label", null,
          h("span", { className: "hermes-workflows-muted" }, "Available context placeholders, one per line"),
          h("textarea", { value: promptAssistantContext, onChange: function (event) { setPromptAssistantContext(event.target.value); } })
        ),
        h("label", null,
          h("span", { className: "hermes-workflows-muted" }, "Expected output contract JSON"),
          h("textarea", { value: promptAssistantOutput, onChange: function (event) { setPromptAssistantOutput(event.target.value); } })
        ),
        h("label", null,
          h("span", { className: "hermes-workflows-muted" }, "Constraints, one per line"),
          h("textarea", { value: promptAssistantConstraints, onChange: function (event) { setPromptAssistantConstraints(event.target.value); } })
        ),
        h("button", { type: "button", onClick: draftPromptWithAssistant, className: "hermes-workflows-primary" }, "Draft prompt")
      );
    }

    function renderAgentCellEditor() {
      return h("div", { className: "hermes-workflows-stack" },
        h("h3", null, "Cell editor"),
        h("div", { className: "hermes-workflows-meta" }, "Agent task " + safeString(selectedNode.id)),
        h("label", null,
          h("span", { className: "hermes-workflows-muted" }, "Assigned profile"),
          h("input", { value: agentProfile, onChange: function (event) { setAgentProfile(event.target.value); }, placeholder: "reviewer" })
        ),
        h("label", null,
          h("span", { className: "hermes-workflows-muted" }, "Task title"),
          h("input", { value: agentTitle, onChange: function (event) { setAgentTitle(event.target.value); }, placeholder: "Review change" })
        ),
        h("label", null,
          h("span", { className: "hermes-workflows-muted" }, "Agent cell prompt"),
          h("textarea", { className: "hermes-workflows-prompt-editor", value: promptText, onChange: function (event) { setPromptText(event.target.value); }, placeholder: "Tell the assigned profile exactly what to do. Use ${ input.foo } or ${ node.previous.output.bar } for workflow context." })
        ),
        h("label", null,
          h("span", { className: "hermes-workflows-muted" }, "Result contract JSON (optional)"),
          h("textarea", { className: "hermes-workflows-contract-editor", value: resultContractText, onChange: function (event) { setResultContractText(event.target.value); } })
        ),
        h("div", { className: "hermes-workflows-row" },
          h("button", { type: "button", onClick: applyAgentCellForm, className: "hermes-workflows-primary" }, "Apply cell prompt"),
          h("button", { type: "button", onClick: function () { setPromptAssistantOpen(!promptAssistantOpen); } }, promptAssistantOpen ? "Hide Prompt assistant" : "Prompt assistant"),
          h("button", { type: "button", onClick: function () { setAdvancedJsonOpen(!advancedJsonOpen); } }, advancedJsonOpen ? "Hide Advanced JSON" : "Advanced JSON")
        ),
        renderPromptAssistant()
      );
    }

    function renderBasicCellEditor() {
      return h("div", { className: "hermes-workflows-stack" },
        h("h3", null, "Cell editor"),
        h("div", { className: "hermes-workflows-meta" }, "Node " + safeString(selectedNode.id)),
        h("p", { className: "hermes-workflows-muted" }, "This node type does not have a prompt form yet. Use Advanced JSON for full node settings."),
        h("div", { className: "hermes-workflows-row" },
          h("button", { type: "button", onClick: function () { setAdvancedJsonOpen(!advancedJsonOpen); } }, advancedJsonOpen ? "Hide Advanced JSON" : "Advanced JSON")
        )
      );
    }

    function renderInspector(spec) {
      return h("aside", { className: "hermes-workflows-inspector" },
        h("h3", null, "Node inspector"),
        selectedNode ? h("div", { className: "hermes-workflows-stack" },
          selectedNode.type === "agent_task" ? renderAgentCellEditor() : renderBasicCellEditor(),
          advancedJsonOpen ? renderAdvancedNodeJson(spec) : null,
          nodeMessage ? h("p", { className: "hermes-workflows-muted" }, nodeMessage) : null
        ) : h("p", { className: "hermes-workflows-muted" }, "Select a node to edit its cell settings. Advanced JSON remains available after selecting a node.")
      );
    }

    function renderReactFlowGraph(spec) {
      if (!ReactFlow || !ReactFlowProvider) return renderSimpleGraph(spec);
      return h("div", { className: "hermes-workflows-builder" },
        h("div", { className: "hermes-workflows-canvas" },
          h(ReactFlowProvider, null,
            h(ReactFlow, {
              nodes: flowNodes,
              edges: flowEdges,
              nodeTypes: NODE_TYPES,
              fitView: true,
              nodesDraggable: true,
              nodesConnectable: true,
              onNodeClick: function (_, node) {
                if (node && node.data && node.data.node) selectNodeForInspector(node.data.node);
              },
              onNodesChange: applyNodeChanges ? function (changes) { setFlowNodes(applyNodeChanges(changes, flowNodes)); } : undefined,
              onEdgesChange: applyEdgeChanges ? function (changes) { setFlowEdges(applyEdgeChanges(changes, flowEdges)); } : undefined,
              onConnect: addEdge ? function (connection) {
                setFlowEdges(addEdge(Object.assign({ label: "draft", markerEnd: { type: MarkerType.ArrowClosed } }, connection), flowEdges));
                const spec = activeSpec();
                if (spec && connection.source && connection.target) {
                  const source = connection.sourceHandle ? connection.source + "." + connection.sourceHandle : connection.source;
                  const nextSpec = upsertSpecEdge(spec, source, connection.target);
                  updateEditorText(specToEditorText(nextSpec));
                  setDraftSpec(nextSpec);
                  setStatus("Connection added to workflow draft.");
                } else {
                  setStatus("Draft connection added visually; validate/select a workflow to persist it.");
                }
              } : undefined,
            },
              Background ? h(Background, null) : null,
              Controls ? h(Controls, null) : null,
              MiniMap ? h(MiniMap, null) : null
            )
          )
        ),
        renderInspector(spec)
      );
    }

    function renderGraph() {
      const spec = activeSpec();
      return h("div", { className: "hermes-workflows-graph" },
        h("div", null,
          h("h2", null, "Visual workflow editor"),
          h("p", { className: "hermes-workflows-muted" }, spec ? safeString(spec.name || spec.id || spec.workflow_id) : "Select or validate a workflow to render its nodes and edges.")
        ),
        spec ? renderReactFlowGraph(spec) : h("p", { className: "hermes-workflows-muted" }, "No workflow graph available yet."),
        !ReactFlow ? h("p", { className: "hermes-workflows-muted" }, "React Flow SDK unavailable; showing the simple HTML graph fallback.") : null
      );
    }

    return h("div", { className: "hermes-workflows" },
      h(Card, { className: "hermes-workflows-header" },
        h("div", null,
          h("h1", null, "Workflows"),
          h("p", { className: "hermes-workflows-muted" }, "Visual workflow builder for definitions, manual runs, executions, and graph inspection.")
        ),
        h("button", { type: "button", disabled: loading, onClick: function () { refresh(); } }, loading ? "Refreshing…" : "Refresh")
      ),
      error ? h("div", { className: "hermes-workflows-banner is-error" }, error) : null,
      status ? h("div", { className: "hermes-workflows-banner" }, status) : null,
      renderGoalBuilder(),
      renderDraftReview(),
      renderValidationChecklist(),
      renderDispatcherReadiness(),
      h("div", { className: "hermes-workflows-grid" },
        h("div", { className: "hermes-workflows-stack" },
          h(Card, { className: "hermes-workflows-panel" },
            h("h2", null, "Workflow list"),
            renderDefinitionList()
          ),
          renderRunInputForm(),
          h(Card, { className: "hermes-workflows-panel" },
            h("h2", null, "Execution list"),
            renderExecutions()
          )
        ),
        h("div", { className: "hermes-workflows-stack" },
          renderAdvancedYaml(),
          h(Card, { className: "hermes-workflows-panel" }, renderGraph()),
          h(Card, { className: "hermes-workflows-panel" },
            h("h2", null, "Execution detail timeline"),
            renderTimeline()
          )
        )
      )
    );
  }

  REG.register("workflows", WorkflowsPage);
})();
