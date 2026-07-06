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
    const stateRunWorkflowId = useState("");
    const runWorkflowId = stateRunWorkflowId[0];
    const setRunWorkflowId = stateRunWorkflowId[1];
    const stateRunInputText = useState("{}");
    const runInputText = stateRunInputText[0];
    const setRunInputText = stateRunInputText[1];
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

    function selectNodeForInspector(node) {
      setSelectedNode(node);
      setNodeJson(jsonBlock(node));
      setNodeMessage("");
      setAdvancedJsonOpen(false);
      const rawPrompt = node ? node.prompt : undefined;
      setAgentProfile(node && node.profile ? String(node.profile) : "");
      setAgentTitle(node && node.title ? String(node.title) : "");
      setPromptText(rawPrompt === null || rawPrompt === undefined ? "" : (typeof rawPrompt === "string" ? rawPrompt : jsonBlock(rawPrompt)));
      setResultContractText(jsonBlock((node && node.result_contract) || {}));
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

    function refresh(preferExecutionId) {
      setLoading(true);
      setError("");
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
      setRunning(true);
      setError("");
      api("/definitions/" + encodeURIComponent(workflowId) + "/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input_json: runInputText }),
      }).then(function (res) {
        const execution = res.execution || {};
        const executionId = execution.execution_id;
        setStatus("Started execution " + safeString(executionId));
        return loadExecutions(executionId);
      }).catch(fail).finally(function () { setRunning(false); });
    }

    function importDefinitionFile(event) {
      const file = event.target.files && event.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = function () {
        updateEditorText(String(reader.result || ""));
        setStatus("Import YAML loaded " + safeString(file.name));
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
        setError("Clipboard API unavailable; use Export YAML instead.");
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
          h("button", { type: "button", disabled: true, title: "Prompt assistant is added in a later task." }, "Prompt assistant"),
          h("button", { type: "button", onClick: function () { setAdvancedJsonOpen(!advancedJsonOpen); } }, advancedJsonOpen ? "Hide Advanced JSON" : "Advanced JSON")
        )
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
      h("div", { className: "hermes-workflows-grid" },
        h("div", { className: "hermes-workflows-stack" },
          h(Card, { className: "hermes-workflows-panel" },
            h("h2", null, "Workflow list"),
            renderDefinitionList()
          ),
          h(Card, { className: "hermes-workflows-panel hermes-workflows-run-form" },
            h("h2", null, "Manual run form"),
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
              h("label", null,
                h("span", { className: "hermes-workflows-muted" }, "Input JSON"),
                h("textarea", {
                  className: "hermes-workflows-run-input",
                  value: runInputText,
                  onChange: function (event) { setRunInputText(event.target.value); },
                })
              ),
              h("button", { type: "submit", disabled: running, className: "hermes-workflows-primary" }, running ? "Running…" : "Run workflow")
            )
          ),
          h(Card, { className: "hermes-workflows-panel" },
            h("h2", null, "Execution list"),
            renderExecutions()
          )
        ),
        h("div", { className: "hermes-workflows-stack" },
          h(Card, { className: "hermes-workflows-panel" },
            h("h2", null, "Validate / deploy definition"),
            h("textarea", {
              className: "hermes-workflows-editor",
              value: editorText,
              onChange: function (event) { updateEditorText(event.target.value); },
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
          ),
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
