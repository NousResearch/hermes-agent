(function() {
	//#region ../plugins/workflows/dashboard/src/graph.js
	function graphItems(spec) {
		const triggers = (spec?.triggers || []).map((trigger, index) => ({
			id: trigger.id || trigger.name || `trigger_${index + 1}`,
			rendererType: "trigger",
			specKind: "trigger",
			triggerType: trigger.type,
			spec: trigger
		}));
		const nodes = Object.entries(spec?.nodes || {}).map(([id, node]) => ({
			id,
			rendererType: node.type,
			specKind: "node",
			triggerType: null,
			spec: node
		}));
		return [...triggers, ...nodes];
	}
	function decorateGraphItems(items, statuses) {
		return items.map((item) => ({
			...item,
			status: statuses[item.id] || "idle"
		}));
	}
	//#endregion
	//#region ../plugins/workflows/dashboard/src/api.js
	function formatApiError(error) {
		if (error instanceof Error && typeof error.message === "string") return formatApiError(error.message);
		if (error && typeof error === "object") {
			if (typeof error.message === "string" && error.message) return error.message;
			if (typeof error.detail === "string" && error.detail) return formatApiError(error.detail);
		}
		if (typeof error === "string") {
			const jsonStart = error.indexOf("{");
			if (jsonStart !== -1) try {
				return formatApiError(JSON.parse(error.slice(jsonStart)));
			} catch {}
			return error;
		}
		return "Unknown error";
	}
	function createApi(fetchJSON, basePath) {
		if (typeof fetchJSON !== "function") throw new Error("createApi requires the SDK fetchJSON implementation");
		const base = String(basePath || "").replace(/\/$/, "");
		return function api(path, options) {
			return fetchJSON(base + String(path || ""), options);
		};
	}
	//#endregion
	//#region ../plugins/workflows/dashboard/src/workspace.js
	var WORKSPACE_MODES = [
		"build",
		"run",
		"history"
	];
	var WORKSPACE_PATH_PREFIX = "/workflows/";
	var MODE_TO_QUERY_KEY = {
		run: "feed",
		history: "execution"
	};
	function isMode(value) {
		return WORKSPACE_MODES.indexOf(value) !== -1;
	}
	function trimSlashes(value) {
		return String(value || "").replace(/^\/+|\/+$/g, "");
	}
	function parseLocation(location) {
		if (!location || typeof location !== "object") return {
			workflowId: "",
			mode: "build",
			search: ""
		};
		const segments = trimSlashes(String(location.pathname || "")).replace(/^\/+/, "").split("/").filter(Boolean);
		let workflowId = "";
		let modeSegment = "";
		if (segments.length >= 2 && segments[0] === "workflows") {
			workflowId = segments[1];
			modeSegment = segments[2] || "";
		}
		const search = String(location.search || "");
		const mode = isMode(modeSegment) ? modeSegment : "build";
		return {
			workflowId,
			mode,
			search
		};
	}
	function modeForLocation(location) {
		return parseLocation(location).mode;
	}
	function selectionForMode(mode, selection) {
		const key = MODE_TO_QUERY_KEY[mode];
		if (!key) return "";
		const value = selection && typeof selection === "object" ? selection[key] : "";
		return value ? String(value) : "";
	}
	function locationForMode(workflowId, mode, selection) {
		const id = trimSlashes(workflowId);
		const safeMode = isMode(mode) ? mode : "build";
		const base = WORKSPACE_PATH_PREFIX + id + (safeMode === "build" ? "" : "/" + safeMode);
		const value = selectionForMode(safeMode, selection);
		return value ? base + "?" + MODE_TO_QUERY_KEY[safeMode] + "=" + encodeURIComponent(value) : base;
	}
	//#endregion
	//#region ../plugins/workflows/dashboard/src/app.js
	(function() {
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
		FlowSDK.Background;
		const Controls = FlowSDK.Controls;
		FlowSDK.MiniMap;
		const Handle = FlowSDK.Handle;
		const Position = FlowSDK.Position || {
			Left: "left",
			Right: "right"
		};
		const MarkerType = FlowSDK.MarkerType || { ArrowClosed: "arrowclosed" };
		const addEdge = FlowSDK.addEdge;
		const applyNodeChanges = FlowSDK.applyNodeChanges;
		const applyEdgeChanges = FlowSDK.applyEdgeChanges;
		const API = "/api/plugins/workflows";
		const DEFINITIONS_API = "/api/plugins/workflows/definitions";
		const NODE_KIND_LIST = [
			"trigger",
			"pass",
			"switch",
			"agent_task",
			"wait",
			"parallel",
			"join",
			"fail"
		];
		const FALLBACK_IMPLEMENTED_TRIGGER_TYPES = ["manual", "schedule"];
		const FALLBACK_IMPLEMENTED_NODE_TYPES = [
			"pass",
			"switch",
			"agent_task",
			"wait",
			"parallel",
			"join",
			"fail"
		];
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
			"edges: []"
		].join("\n");
		const api = createApi(SDK.fetchJSON, API);
		function errorMessage(err) {
			return formatApiError(err);
		}
		const PRIVACY_NOTE = "Workflow inputs and outputs are stored locally in Hermes workflow/Kanban history. Do not paste secrets; common secret-looking keys are redacted in dashboard views.";
		const SCALAR_INPUT_KINDS = [
			"text",
			"long_text",
			"prompt",
			"criteria",
			"url",
			"repo_path",
			"boolean",
			"number",
			"integer"
		];
		const INTAKE_SCOPE_NOTE = "Phase 1 supports scalar manual and continuous input items. Batch splitting and document uploads are not supported in this release.";
		function asArray(value) {
			return Array.isArray(value) ? value : [];
		}
		function safeString(value) {
			if (value === null || value === void 0 || value === "") return "—";
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
		function hasPreviewValue(value) {
			if (value === null || value === void 0) return false;
			if (typeof value === "string") return !!value.trim();
			if (Array.isArray(value)) return value.length > 0;
			if (typeof value === "object") return Object.keys(value).length > 0;
			return true;
		}
		function previewJson(value) {
			let text;
			try {
				text = JSON.stringify(value, null, 2);
			} catch (_) {
				text = String(value);
			}
			if (text === void 0) text = String(value);
			return text.length > 800 ? text.slice(0, 800) + "\n…" : text;
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
		function currentLocationShape() {
			if (typeof window === "undefined" || !window.location) return {
				pathname: "",
				search: ""
			};
			return {
				pathname: window.location.pathname || "",
				search: window.location.search || ""
			};
		}
		function workflowIdFromLocation() {
			if (typeof window === "undefined" || !window.location) return "";
			const match = String(window.location.pathname || "").match(/^\/workflows\/([^\/]+)/);
			return match ? decodeURIComponent(match[1]) : "";
		}
		function pushMode(nextMode, selection) {
			if (typeof window === "undefined" || !window.history) return;
			if (nextMode === modeForLocation(currentLocationShape())) return;
			const target = locationForMode(workflowIdFromLocation() || "__draft__", nextMode, selection || {});
			if (typeof window.history.pushState === "function") window.history.pushState({ workspaceMode: nextMode }, "", target);
			else window.location.href = target;
		}
		function nodeList(spec) {
			const nodes = spec && spec.nodes ? spec.nodes : {};
			if (Array.isArray(nodes)) return nodes.map(function(node, index) {
				return Object.assign({
					id: node.id || node.name || "node_" + (index + 1),
					specKind: "node"
				}, node || {});
			});
			return Object.keys(nodes).map(function(id) {
				return Object.assign({
					id,
					specKind: "node"
				}, nodes[id] || {});
			});
		}
		function isTriggerSource(spec, sourceId) {
			const requested = String(sourceId || "");
			return graphItems(spec).some(function(item) {
				return item.specKind === "trigger" && item.id === requested;
			});
		}
		function splitPort(value) {
			const parts = safeString(value).split(".");
			return {
				nodeId: parts.shift() || "?",
				port: parts.join(".")
			};
		}
		function edgeList(spec) {
			return asArray(spec && spec.edges).map(function(edge, index) {
				const source = splitPort(edge.from || edge.source || edge.start || "?");
				const target = splitPort(edge.to || edge.target || edge.end || "?");
				return {
					id: edge.id || String(index + 1),
					from: source.nodeId,
					to: target.nodeId,
					label: edge.label || edge.condition || source.port || target.port || "",
					raw: edge
				};
			});
		}
		function promptObjectiveText(prompt) {
			if (typeof prompt === "string") return prompt;
			if (Array.isArray(prompt)) return prompt.map(promptObjectiveText).filter(Boolean).join("; ");
			if (prompt && typeof prompt === "object") return prompt.task || prompt.objective || prompt.title || prompt.goal || prompt.description || "";
			return "";
		}
		function inputFieldsForSpec(spec, preferredTrigger) {
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
					if ({
						text: true,
						long_text: true,
						document: true,
						prompt: true,
						criteria: true,
						url: true,
						repo_path: true,
						boolean: true,
						number: true,
						integer: true,
						object: true
					}[type]) return type === "object" ? "json" : type;
					if (["float"].indexOf(type) !== -1) return "number";
					if (["bool"].indexOf(type) !== -1) return "boolean";
					if (["array"].indexOf(type) !== -1) return "json";
					if (type === "string") return "text";
					if (value.default !== void 0) return kindForLiteral(value.default);
					if (value.example !== void 0) return kindForLiteral(value.example);
					return "json";
				}
				return kindForLiteral(value);
			}
			function triggerInputSchemaObject(trigger) {
				const schema = trigger && trigger.input_schema;
				if (!schema || typeof schema !== "object" || Array.isArray(schema)) return {};
				return schema;
			}
			function triggerInputObject(trigger) {
				const rawInput = trigger && trigger.input;
				if (!rawInput || typeof rawInput !== "object" || Array.isArray(rawInput)) return {};
				if (rawInput.type === "object") {
					if (rawInput.properties && typeof rawInput.properties === "object" && !Array.isArray(rawInput.properties)) return rawInput.properties;
					const schemaMetadataKeys = {
						type: true,
						required: true,
						additionalProperties: true,
						description: true,
						title: true,
						$schema: true
					};
					if (Object.keys(rawInput).every(function(key) {
						return schemaMetadataKeys[key];
					})) return {};
				}
				return rawInput;
			}
			function fieldsFromInputSchema(schema) {
				return Object.keys(schema).map(function(key) {
					const field = schema[key] && typeof schema[key] === "object" && !Array.isArray(schema[key]) ? schema[key] : {};
					return {
						name: key,
						kind: kindForInputValue(field),
						label: String(field.label || key),
						required: !!field.required,
						description: String(field.description || "")
					};
				});
			}
			const preferredSchema = triggerInputSchemaObject(preferredTrigger);
			if (Object.keys(preferredSchema).length) return fieldsFromInputSchema(preferredSchema);
			const triggers = asArray(spec && spec.triggers);
			const schemaTrigger = triggers.find(function(trigger) {
				if (!trigger || typeof trigger !== "object") return false;
				return String(trigger.type || trigger.trigger_type || "") === "manual" && Object.keys(triggerInputSchemaObject(trigger)).length;
			}) || triggers.find(function(trigger) {
				return Object.keys(triggerInputSchemaObject(trigger)).length;
			});
			if (schemaTrigger) return fieldsFromInputSchema(triggerInputSchemaObject(schemaTrigger));
			const inputTrigger = triggers.find(function(trigger) {
				if (!trigger || typeof trigger !== "object") return false;
				return String(trigger.type || trigger.trigger_type || "") === "manual" && Object.keys(triggerInputObject(trigger)).length;
			}) || triggers.find(function(trigger) {
				return Object.keys(triggerInputObject(trigger)).length;
			});
			const triggerInput = inputTrigger ? triggerInputObject(inputTrigger) : {};
			const keys = Object.keys(triggerInput);
			if (keys.length) return keys.map(function(key) {
				return {
					name: key,
					kind: kindForInputValue(triggerInput[key])
				};
			});
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
			text.replace(/\$\{\s*input\.([a-zA-Z0-9_\-.\[\]]+)\s*\}/g, function(_, key) {
				addFallbackInputField(key);
				return "";
			});
			text.replace(/\$\.input\.([a-zA-Z0-9_\-.\[\]]+)/g, function(_, key) {
				addFallbackInputField(key);
				return "";
			});
			return Object.keys(found).sort().map(function(name) {
				return {
					name,
					kind: found[name]
				};
			});
		}
		function inputRowsFromTrigger(trigger) {
			const schema = trigger && trigger.input_schema && typeof trigger.input_schema === "object" && !Array.isArray(trigger.input_schema) ? trigger.input_schema : {};
			return Object.keys(schema).sort().map(function(name) {
				const field = schema[name] && typeof schema[name] === "object" && !Array.isArray(schema[name]) ? schema[name] : {};
				let kind = String(field.kind || "text");
				if (kind === "object" || kind === "json" || kind === "document") kind = "text";
				if (SCALAR_INPUT_KINDS.indexOf(kind) === -1) kind = "text";
				return {
					name,
					kind,
					required: !!field.required,
					defaultValue: field.default === void 0 || field.default === null ? "" : String(field.default),
					minLength: field.min_length === void 0 || field.min_length === null ? "" : String(field.min_length),
					maxLength: field.max_length === void 0 || field.max_length === null ? "" : String(field.max_length)
				};
			});
		}
		function inputSchemaFromRows(rows) {
			const schema = {};
			asArray(rows).forEach(function(row) {
				const name = workflowIdFromText(row && row.name ? row.name : "").replace(/-/g, "_");
				if (!name) return;
				const field = { kind: SCALAR_INPUT_KINDS.indexOf(row.kind) !== -1 ? row.kind : "text" };
				if (row.required) field.required = true;
				if (row.defaultValue !== void 0 && String(row.defaultValue).trim() !== "") field.default = String(row.defaultValue);
				const minLength = String(row.minLength || "").trim();
				const maxLength = String(row.maxLength || "").trim();
				if (minLength) field.min_length = Math.max(0, parseInt(minLength, 10) || 0);
				if (maxLength) field.max_length = Math.max(0, parseInt(maxLength, 10) || 0);
				schema[name] = field;
			});
			return schema;
		}
		function readyPathFromTrigger(trigger) {
			const cond = trigger && trigger.intake && trigger.intake.ready_when;
			if (!cond || typeof cond !== "object" || Array.isArray(cond)) return "";
			if (String(cond.op || "") === "exists" && typeof cond.path === "string") return cond.path;
			return "";
		}
		function triggerIntakeFromForm(mode, dedupeKey, readyPath) {
			const intake = { mode: mode === "continuous" ? "continuous" : "single" };
			const dedupe = String(dedupeKey || "").trim();
			const ready = String(readyPath || "").trim();
			if (dedupe) intake.dedupe_key = dedupe;
			if (ready) intake.ready_when = {
				op: "exists",
				path: ready
			};
			return intake;
		}
		function inputObjectForFields(fields, values) {
			const input = {};
			asArray(fields).forEach(function(field) {
				const name = field && field.name;
				if (!name) return;
				const raw = values && values[name];
				if (raw === void 0 || raw === null || raw === "") return;
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
			if (nodes && typeof nodes === "object") return Object.keys(nodes).map(function(id) {
				return nodes[id] && typeof nodes[id] === "object" ? nodes[id] : {};
			});
			return [];
		}
		function checklistNodesShapeValid(spec) {
			const nodes = spec && spec.nodes;
			if (Array.isArray(nodes)) return false;
			if (nodes && typeof nodes === "object") return Object.keys(nodes).every(function(id) {
				const node = nodes[id];
				return node && typeof node === "object" && !Array.isArray(node);
			});
			return false;
		}
		function checklistNodeIdsValid(spec) {
			const nodes = spec && spec.nodes;
			if (!nodes || typeof nodes !== "object" || Array.isArray(nodes)) return false;
			return Object.keys(nodes).every(function(id) {
				return /^[a-z][a-z0-9_-]{0,63}$/.test(id);
			});
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
			return triggers.every(function(trigger) {
				if (!trigger || typeof trigger !== "object" || Array.isArray(trigger)) return false;
				if (typeof trigger.type !== "string") return false;
				return implementedTriggers.indexOf(trigger.type) !== -1;
			});
		}
		function checklistNodesImplemented(nodes, capabilities) {
			const implementedNodes = implementedNodeTypesFromCapabilities(capabilities);
			return !nodes.some(function(node) {
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
			return edges.every(function(edge) {
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
				if (branch !== null && (!branch || sourceType !== "switch" && sourceType !== "parallel")) return false;
				outgoingSources[sourceBase] = true;
			}
			return Object.keys(nodes).every(function(nodeId) {
				const node = nodes[nodeId];
				if (node.catch !== null && node.catch !== void 0) {
					if (typeof node.catch !== "string" || node.catch === nodeId || !hasNode(node.catch)) return false;
				}
				if (node.type === "switch") {
					if (node.default !== null && node.default !== void 0) return typeof node.default === "string" && hasNode(node.default);
					return !!outgoingSources[nodeId];
				}
				return true;
			});
		}
		function validationChecklist(spec, capabilities) {
			const safeNodes = checklistNodeValues(spec);
			return [
				{
					label: "Workflow id set",
					ok: checklistWorkflowIdValid(spec && spec.id)
				},
				{
					label: "Workflow name set",
					ok: hasStringValue(spec && spec.name)
				},
				{
					label: "Version set",
					ok: checklistVersionValid(spec && spec.version)
				},
				{
					label: "At least one node",
					ok: safeNodes.length > 0
				},
				{
					label: "Node definitions are objects",
					ok: checklistNodesShapeValid(spec)
				},
				{
					label: "Node ids are valid",
					ok: checklistNodeIdsValid(spec)
				},
				{
					label: "No unsupported triggers (implemented today)",
					ok: checklistTriggersImplemented(spec, capabilities)
				},
				{
					label: "No unsupported nodes (implemented today)",
					ok: checklistNodesImplemented(safeNodes, capabilities)
				},
				{
					label: "Edges refer to known nodes",
					ok: checklistEdgesReferToKnownNodes(spec)
				},
				{
					label: "Graph rules pass",
					ok: checklistGraphRulesValid(spec)
				},
				{
					label: "Agent cells have profile and prompt",
					ok: !safeNodes.some(function(node) {
						return node.type === "agent_task" && (!hasProfileValue(node.profile) || !hasPromptValue(node.prompt));
					})
				}
			];
		}
		function eventStatus(row) {
			const payload = row && row.payload || {};
			const kind = String(row && row.kind || "");
			if (payload.status) return payload.status;
			if (kind.indexOf("node_succeeded") !== -1) return "succeeded";
			if (kind.indexOf("node_failed") !== -1) return "failed";
			if (kind.indexOf("execution_blocked") !== -1) return "blocked";
			if (kind.indexOf("execution_waiting") !== -1) return "waiting";
			return "";
		}
		function statusByNode(events) {
			const statuses = {};
			asArray(events).forEach(function(row) {
				const payload = row && row.payload || {};
				const status = eventStatus(row);
				asArray(payload.waiting_nodes).forEach(function(nodeId) {
					statuses[nodeId] = "waiting";
				});
				const nodeId = payload.node_id || payload.nodeId || payload.node || payload.error && payload.error.node || row.node_id;
				if (nodeId && status) statuses[nodeId] = status;
			});
			return statuses;
		}
		function makeWorkflowNode(kind) {
			return function WorkflowNode(props) {
				const data = props && props.data || {};
				const status = data.status || "idle";
				const node = data.node || {};
				return h("div", {
					className: "hermes-workflows-rf-node is-" + classSafe(kind) + " is-status-" + classSafe(status),
					role: "button",
					tabIndex: 0,
					"aria-label": "Edit workflow cell " + safeString(data.id || node.id || "cell"),
					onClick: function(event) {
						event.stopPropagation();
						if (data.onSelect) data.onSelect(node);
					},
					onKeyDown: function(event) {
						if (event.key === "Enter" || event.key === " ") {
							event.preventDefault();
							event.stopPropagation();
							if (data.onSelect) data.onSelect(node);
						}
					}
				}, Handle ? h(Handle, {
					type: "target",
					position: Position.Left
				}) : null, h("div", { className: "hermes-workflows-rf-node-title" }, safeString(node.id || data.id)), h("div", { className: "hermes-workflows-rf-node-type" }, kind), status && status !== "idle" ? h("div", { className: "hermes-workflows-rf-node-status" }, safeString(status)) : null, Handle ? h(Handle, {
					type: "source",
					position: Position.Right
				}) : null);
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
			fail: makeWorkflowNode("fail")
		};
		function buildFlowNodes(spec, statuses, selectedNode, onSelect, nodePositions) {
			return decorateGraphItems(graphItems(spec), statuses || {}).map(function(item, index) {
				const id = item.id || item.spec.id || item.spec.name || "node_" + (index + 1);
				const rendererType = item.rendererType;
				const kind = NODE_TYPES[rendererType] ? rendererType : "pass";
				const pos = (nodePositions || {})[id] || {
					x: index % 3 * 250,
					y: Math.floor(index / 3) * 155
				};
				const legacyNode = item.specKind === "trigger" ? Object.assign({}, item.spec, {
					id,
					specKind: "trigger",
					type: "trigger",
					trigger_type: item.triggerType
				}) : Object.assign({}, item.spec, {
					id,
					specKind: "node",
					type: rendererType
				});
				return {
					id,
					type: kind,
					position: pos,
					className: "hermes-workflows-rf-node-shell is-status-" + classSafe(item.status) + (selectedNode && selectedNode.id === id ? " is-selected" : ""),
					data: {
						id,
						node: legacyNode,
						status: item.status,
						onSelect
					}
				};
			});
		}
		function buildFlowEdges(spec) {
			return edgeList(spec).map(function(edge) {
				return {
					id: edge.id,
					source: edge.from,
					target: edge.to,
					label: edge.label,
					markerEnd: { type: MarkerType.ArrowClosed }
				};
			});
		}
		function cleanedNodeForSpec(node) {
			const cleaned = Object.assign({}, node || {});
			const providerText = providerValue(cleaned).trim();
			const modelText = modelValue(cleaned).trim();
			delete cleaned.specKind;
			delete cleaned.trigger_type;
			delete cleaned.provider_override;
			delete cleaned.model_override;
			delete cleaned.provider;
			delete cleaned.model;
			if (cleaned.type === "agent_task") {
				if (providerText) cleaned.provider = providerText;
				if (modelText) cleaned.model = modelText;
			} else {
				delete cleaned.profile;
				delete cleaned.result_contract;
				delete cleaned.skills;
				delete cleaned.workspace_kind;
				delete cleaned.workspace_path;
				delete cleaned.goal_mode;
				delete cleaned.goal_max_turns;
				delete cleaned.max_retries;
			}
			if (cleaned.type !== "switch") {
				delete cleaned.cases;
				delete cleaned.default;
			}
			if (cleaned.type !== "wait") delete cleaned.seconds;
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
			if (Array.isArray(spec.nodes)) return spec.nodes.find(function(node) {
				return (node.id || node.name) === nodeId;
			}) || null;
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
				const sourceIndex = next.nodes.findIndex(function(node) {
					return (node && (node.id || node.name)) === nodeId;
				});
				const targetIndex = next.nodes.findIndex(function(node) {
					return (node && (node.id || node.name)) === nextId;
				});
				const index = sourceIndex >= 0 ? sourceIndex : targetIndex;
				if (index >= 0) {
					next.nodes[index] = nodeWithId;
					next.nodes = next.nodes.filter(function(node, nodeIndex) {
						const id = node && (node.id || node.name);
						return nodeIndex === index || id !== nextId;
					});
				} else next.nodes.push(nodeWithId);
			} else {
				next.nodes = next.nodes || {};
				if (nodeId && nodeId !== nextId) delete next.nodes[nodeId];
				next.nodes[nextId] = clean;
			}
			if (nodeId && nodeId !== nextId) rewriteSpecNodeReferences(next, nodeId, nextId);
			return next;
		}
		function rewriteSpecNodeReferences(spec, oldId, newId) {
			const oldText = String(oldId || "");
			const newText = String(newId || "");
			if (!oldText || !newText || oldText === newText) return spec;
			spec.edges = asArray(spec.edges).map(function(edge) {
				const nextEdge = Object.assign({}, edge || {});
				["from", "source"].forEach(function(key) {
					if (typeof nextEdge[key] !== "string") return;
					if (nextEdge[key] === oldText) nextEdge[key] = newText;
					else if (nextEdge[key].indexOf(oldText + ".") === 0) nextEdge[key] = newText + nextEdge[key].slice(oldText.length);
				});
				["to", "target"].forEach(function(key) {
					if (nextEdge[key] === oldText) nextEdge[key] = newText;
				});
				return nextEdge;
			});
			Object.keys(spec.nodes || {}).forEach(function(key) {
				const node = spec.nodes[key];
				if (!node || typeof node !== "object") return;
				if (node.default === oldText) node.default = newText;
				if (node.catch === oldText) node.catch = newText;
			});
			return spec;
		}
		function upsertSpecEdge(spec, source, target) {
			const next = cloneSpec(spec);
			next.edges = asArray(next.edges);
			if (!next.edges.some(function(edge) {
				return (edge.from || edge.source) === source && (edge.to || edge.target) === target;
			}) && source && target) next.edges.push({
				from: source,
				to: target
			});
			return next;
		}
		function workflowIdFromText(value) {
			const slug = String(value || "workflow draft").trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 64);
			if (slug && /^[a-z]/.test(slug)) return slug;
			return "workflow_draft";
		}
		function nodeIdsForSpec(spec) {
			return nodeList(spec).map(function(node) {
				return String(node.id || node.name || "");
			}).filter(Boolean);
		}
		function uniqueWorkflowId(spec, requested) {
			const existing = Object.create(null);
			nodeIdsForSpec(spec).forEach(function(id) {
				existing[id] = true;
			});
			asArray(spec && spec.triggers).forEach(function(trigger) {
				const id = trigger && (trigger.id || trigger.name);
				if (id) existing[String(id)] = true;
			});
			const base = workflowIdFromText(requested || "cell").replace(/_+/g, "_") || "cell";
			let candidate = base;
			let index = 2;
			while (existing[candidate]) {
				candidate = (base + "_" + index).slice(0, 64);
				index += 1;
			}
			return candidate;
		}
		function newWorkflowSpec(name) {
			const title = String(name || "Untitled Workflow").trim() || "Untitled Workflow";
			return {
				id: workflowIdFromText(title),
				name: title,
				version: 1,
				enabled: true,
				triggers: [{
					id: "manual",
					type: "manual"
				}],
				nodes: {},
				edges: []
			};
		}
		function defaultNodeForType(type, nodeId) {
			const safeType = NODE_KIND_LIST.indexOf(type) === -1 || type === "trigger" ? "pass" : type;
			const node = {
				type: safeType,
				title: String(nodeId || safeType || "cell").replace(/[_-]+/g, " ").trim() || safeType
			};
			if (safeType === "agent_task") {
				node.profile = "default";
				node.prompt = "Return JSON only matching the result contract.";
				node.result_contract = {
					summary: "string",
					status: "string"
				};
			} else if (safeType === "wait") node.seconds = 60;
			else if (safeType === "fail") node.output = "Workflow failed.";
			else if (safeType === "switch") node.cases = [];
			return node;
		}
		function addSpecNodeAfter(spec, nodeId, type, afterSource) {
			const next = cloneSpec(spec || newWorkflowSpec("Workflow Draft"));
			next.nodes = next.nodes || {};
			next.edges = asArray(next.edges);
			const id = uniqueWorkflowId(next, nodeId || type || "cell");
			const clean = defaultNodeForType(type || "pass", id);
			if (Array.isArray(next.nodes)) next.nodes.push(Object.assign({ id }, clean));
			else next.nodes[id] = clean;
			const source = String(afterSource || "").trim();
			if (source) return upsertSpecEdge(next, source, id);
			return next;
		}
		function addSpecTrigger(spec, triggerId, triggerType, scheduleText) {
			const next = cloneSpec(spec || newWorkflowSpec("Workflow Draft"));
			const type = triggerType === "schedule" ? "schedule" : "manual";
			const id = uniqueWorkflowId(next, triggerId || type || "trigger");
			const trigger = {
				id,
				type
			};
			if (type === "schedule") trigger.schedule = String(scheduleText || "0 9 * * *").trim() || "0 9 * * *";
			next.triggers = asArray(next.triggers).filter(function(existing) {
				return String(existing && (existing.id || existing.name) || "") !== id;
			});
			next.triggers.push(trigger);
			return next;
		}
		function addSwitchCaseToSpec(spec, nodeId, caseName, path, equalsValue) {
			const next = cloneSpec(spec || {});
			const id = String(nodeId || "");
			const name = workflowIdFromText(caseName || "case");
			const comparePath = String(path || "$.input.status").trim() || "$.input.status";
			if (!id || !next.nodes || !next.nodes[id]) return next;
			const node = Object.assign({}, next.nodes[id], { type: "switch" });
			const caseSpec = {
				name,
				when: {
					op: "eq",
					left: { path: comparePath },
					right: String(equalsValue || name)
				}
			};
			node.cases = asArray(node.cases).filter(function(item) {
				return item && item.name !== name;
			}).concat([caseSpec]);
			next.nodes[id] = node;
			return next;
		}
		function removeSpecNode(spec, nodeId) {
			const next = cloneSpec(spec || {});
			const id = String(nodeId || "");
			if (!id) return next;
			if (Array.isArray(next.nodes)) next.nodes = next.nodes.filter(function(node) {
				return String(node && (node.id || node.name) || "") !== id;
			});
			else if (next.nodes && typeof next.nodes === "object") delete next.nodes[id];
			next.edges = asArray(next.edges).filter(function(edge) {
				const sourceBase = String(edge && (edge.from || edge.source) || "").split(".", 1)[0];
				const target = String(edge && (edge.to || edge.target) || "");
				return sourceBase !== id && target !== id;
			});
			Object.keys(next.nodes || {}).forEach(function(key) {
				const node = next.nodes[key];
				if (!node || typeof node !== "object") return;
				if (node.default === id) delete node.default;
				if (node.catch === id) delete node.catch;
			});
			return next;
		}
		function removeSpecTrigger(spec, triggerId) {
			const next = cloneSpec(spec || {});
			const id = String(triggerId || "");
			next.triggers = asArray(next.triggers).filter(function(trigger) {
				return String(trigger && (trigger.id || trigger.name) || "") !== id;
			});
			return next;
		}
		function providerValue(node) {
			return node && (node.provider || node.provider_override) ? String(node.provider || node.provider_override) : "";
		}
		function modelValue(node) {
			return node && (node.model || node.model_override) ? String(node.model || node.model_override) : "";
		}
		function providerRows(options) {
			return asArray(options && options.providers).filter(function(provider) {
				return provider && (provider.slug || provider.provider);
			});
		}
		function profileRows(options) {
			return asArray(options && options.profiles).filter(function(profile) {
				return profile && profile.name;
			});
		}
		function WorkflowsPage() {
			const useState = React.useState;
			const useEffect = React.useEffect;
			const useRef = React.useRef;
			const flowInstanceRef = useRef(null);
			const membershipKeyRef = useRef("");
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
			const stateAgentProvider = useState("");
			const agentProvider = stateAgentProvider[0];
			const setAgentProvider = stateAgentProvider[1];
			const stateAgentModel = useState("");
			const agentModel = stateAgentModel[0];
			const setAgentModel = stateAgentModel[1];
			const stateAgentRoutingOptions = useState({
				profiles: [],
				providers: [],
				default_provider: "",
				default_model: ""
			});
			const agentRoutingOptions = stateAgentRoutingOptions[0];
			const setAgentRoutingOptions = stateAgentRoutingOptions[1];
			const stateAgentTitle = useState("");
			const agentTitle = stateAgentTitle[0];
			const setAgentTitle = stateAgentTitle[1];
			const stateCellId = useState("");
			const cellId = stateCellId[0];
			const setCellId = stateCellId[1];
			const stateCellType = useState("pass");
			const cellType = stateCellType[0];
			const setCellType = stateCellType[1];
			const stateTriggerInputRows = useState([]);
			const triggerInputRows = stateTriggerInputRows[0];
			const setTriggerInputRows = stateTriggerInputRows[1];
			const stateTriggerInputName = useState("");
			const triggerInputName = stateTriggerInputName[0];
			const setTriggerInputName = stateTriggerInputName[1];
			const stateTriggerInputKind = useState("text");
			const triggerInputKind = stateTriggerInputKind[0];
			const setTriggerInputKind = stateTriggerInputKind[1];
			const stateTriggerInputRequired = useState(true);
			const triggerInputRequired = stateTriggerInputRequired[0];
			const setTriggerInputRequired = stateTriggerInputRequired[1];
			const stateTriggerInputDefault = useState("");
			const triggerInputDefault = stateTriggerInputDefault[0];
			const setTriggerInputDefault = stateTriggerInputDefault[1];
			const stateTriggerInputMinLength = useState("");
			const triggerInputMinLength = stateTriggerInputMinLength[0];
			const setTriggerInputMinLength = stateTriggerInputMinLength[1];
			const stateTriggerInputMaxLength = useState("");
			const triggerInputMaxLength = stateTriggerInputMaxLength[0];
			const setTriggerInputMaxLength = stateTriggerInputMaxLength[1];
			const stateTriggerIntakeMode = useState("single");
			const triggerIntakeMode = stateTriggerIntakeMode[0];
			const setTriggerIntakeMode = stateTriggerIntakeMode[1];
			const stateTriggerDedupeKey = useState("");
			const triggerDedupeKey = stateTriggerDedupeKey[0];
			const setTriggerDedupeKey = stateTriggerDedupeKey[1];
			const stateTriggerReadyPath = useState("");
			const triggerReadyPath = stateTriggerReadyPath[0];
			const setTriggerReadyPath = stateTriggerReadyPath[1];
			const stateTriggerSchedule = useState("");
			const triggerSchedule = stateTriggerSchedule[0];
			const setTriggerSchedule = stateTriggerSchedule[1];
			const stateCellOutputText = useState("");
			const cellOutputText = stateCellOutputText[0];
			const setCellOutputText = stateCellOutputText[1];
			const stateCellSeconds = useState("60");
			const cellSeconds = stateCellSeconds[0];
			const setCellSeconds = stateCellSeconds[1];
			const stateSwitchDefault = useState("");
			const switchDefault = stateSwitchDefault[0];
			const setSwitchDefault = stateSwitchDefault[1];
			const stateSwitchCases = useState([]);
			const switchCases = stateSwitchCases[0];
			const setSwitchCases = stateSwitchCases[1];
			const stateSwitchCaseName = useState("");
			const switchCaseName = stateSwitchCaseName[0];
			const setSwitchCaseName = stateSwitchCaseName[1];
			const stateSwitchCasePath = useState("$.input.status");
			const switchCasePath = stateSwitchCasePath[0];
			const setSwitchCasePath = stateSwitchCasePath[1];
			const stateSwitchCaseEquals = useState("");
			const switchCaseEquals = stateSwitchCaseEquals[0];
			const setSwitchCaseEquals = stateSwitchCaseEquals[1];
			const stateNewWorkflowName = useState("");
			const newWorkflowName = stateNewWorkflowName[0];
			const setNewWorkflowName = stateNewWorkflowName[1];
			const stateNewTriggerSchedule = useState("0 9 * * *");
			const newTriggerSchedule = stateNewTriggerSchedule[0];
			stateNewTriggerSchedule[1];
			const stateAdvancedJsonOpen = useState(false);
			const advancedJsonOpen = stateAdvancedJsonOpen[0];
			const setAdvancedJsonOpen = stateAdvancedJsonOpen[1];
			const stateIsDragOver = useState(false);
			const isDragOver = stateIsDragOver[0];
			const setIsDragOver = stateIsDragOver[1];
			const stateNodePositions = useState({});
			const nodePositions = stateNodePositions[0];
			const setNodePositions = stateNodePositions[1];
			const stateContextMenu = useState({
				x: 0,
				y: 0,
				visible: false
			});
			const contextMenu = stateContextMenu[0];
			const setContextMenu = stateContextMenu[1];
			const statePromptAssistantOpen = useState(false);
			const promptAssistantOpen = statePromptAssistantOpen[0];
			const setPromptAssistantOpen = statePromptAssistantOpen[1];
			const statePromptAssistantAdvanced = useState(false);
			const promptAssistantAdvanced = statePromptAssistantAdvanced[0];
			const setPromptAssistantAdvanced = statePromptAssistantAdvanced[1];
			const statePromptAssistantGoal = useState("");
			const promptAssistantGoal = statePromptAssistantGoal[0];
			const setPromptAssistantGoal = statePromptAssistantGoal[1];
			const statePromptAssistantObjective = useState("");
			const promptAssistantObjective = statePromptAssistantObjective[0];
			const setPromptAssistantObjective = statePromptAssistantObjective[1];
			const statePromptAssistantContext = useState("${ input }\n${ node.previous.output }");
			const promptAssistantContext = statePromptAssistantContext[0];
			const setPromptAssistantContext = statePromptAssistantContext[1];
			const statePromptAssistantOutput = useState("{\"summary\":\"string\",\"status\":\"string\"}");
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
			stateDraftResult[0];
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
			const stateWorkspaceMode = useState(modeForLocation(currentLocationShape()));
			const workspaceMode = stateWorkspaceMode[0];
			const setWorkspaceMode = stateWorkspaceMode[1];
			const stateDiagnosticsOpen = useState(false);
			const diagnosticsOpen = stateDiagnosticsOpen[0];
			const setDiagnosticsOpen = stateDiagnosticsOpen[1];
			const stateBottomCollapsed = useState(true);
			const bottomCollapsed = stateBottomCollapsed[0];
			const setBottomCollapsed = stateBottomCollapsed[1];
			const stateSidebarCollapsed = useState({});
			const sidebarCollapsed = stateSidebarCollapsed[0];
			const setSidebarCollapsed = stateSidebarCollapsed[1];
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
			const stateRunPanelOpen = useState(false);
			const runPanelOpen = stateRunPanelOpen[0];
			const setRunPanelOpen = stateRunPanelOpen[1];
			const stateInputFeeds = useState([]);
			const inputFeeds = stateInputFeeds[0];
			const setInputFeeds = stateInputFeeds[1];
			const stateInputFeedItems = useState([]);
			const inputFeedItems = stateInputFeedItems[0];
			const setInputFeedItems = stateInputFeedItems[1];
			const stateSelectedFeedId = useState("");
			const selectedFeedId = stateSelectedFeedId[0];
			const setSelectedFeedId = stateSelectedFeedId[1];
			const stateFeedInputValues = useState({});
			const feedInputValues = stateFeedInputValues[0];
			const setFeedInputValues = stateFeedInputValues[1];
			const stateFeedInputText = useState("{}");
			const feedInputText = stateFeedInputText[0];
			const setFeedInputText = stateFeedInputText[1];
			const stateShowAdvancedFeedInputJson = useState(false);
			const showAdvancedFeedInputJson = stateShowAdvancedFeedInputJson[0];
			const setShowAdvancedFeedInputJson = stateShowAdvancedFeedInputJson[1];
			const stateFeedBusy = useState(false);
			const feedBusy = stateFeedBusy[0];
			const setFeedBusy = stateFeedBusy[1];
			const stateEvents = useState([]);
			const events = stateEvents[0];
			const setEvents = stateEvents[1];
			const stateNodeRuns = useState([]);
			const nodeRuns = stateNodeRuns[0];
			const setNodeRuns = stateNodeRuns[1];
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
			const stateDeleting = useState(false);
			const deleting = stateDeleting[0];
			const setDeleting = stateDeleting[1];
			const stateRunning = useState(false);
			const running = stateRunning[0];
			const setRunning = stateRunning[1];
			const stateTicking = useState(false);
			const ticking = stateTicking[0];
			const setTicking = stateTicking[1];
			const initialExecutionId = initialExecutionIdFromLocation();
			function fail(err) {
				setError(errorMessage(err));
			}
			function clearBanners() {
				setError("");
				setStatus("");
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
			function versionForSpec(spec) {
				return spec && spec.version !== void 0 && spec.version !== null ? spec.version : null;
			}
			function versionQuery(version) {
				const value = version === void 0 || version === null ? "" : String(version);
				return value ? "?version=" + encodeURIComponent(value) : "";
			}
			function workflowIdForDefinition(definition) {
				return definition && String(definition.workflow_id || definition.id || "");
			}
			function versionForDefinition(definition) {
				return definition && definition.version !== void 0 && definition.version !== null ? String(definition.version) : "";
			}
			function definitionSelectionKey(definition) {
				const id = workflowIdForDefinition(definition);
				return id ? id + ":" + versionForDefinition(definition) : "";
			}
			function runInputSpec() {
				const selectedId = workflowIdForDefinition(selectedDefinition);
				if (selectedDefinition && selectedId === String(runWorkflowId || "")) return selectedDefinition.spec || null;
				const spec = activeSpec();
				if (spec && workflowIdForSpec(spec) === String(runWorkflowId || "")) return spec;
				return null;
			}
			function selectedInputTrigger(spec) {
				return asArray(spec && spec.triggers).find(function(trigger) {
					return trigger && String(trigger.type || trigger.trigger_type || "") === "manual" && trigger.intake && trigger.intake.mode === "continuous";
				}) || null;
			}
			function loadInputFeedItems(feedId) {
				if (!feedId) {
					setInputFeedItems([]);
					return Promise.resolve([]);
				}
				return api("/input-feeds/" + encodeURIComponent(feedId) + "/items").then(function(res) {
					const rows = asArray(res.items);
					setInputFeedItems(rows);
					return rows;
				}).catch(function() {
					setInputFeedItems([]);
					return [];
				});
			}
			function loadInputFeeds(workflowId) {
				const id = workflowId || workflowIdForDefinition(selectedDefinition);
				if (!id) {
					setInputFeeds([]);
					setSelectedFeedId("");
					setInputFeedItems([]);
					return Promise.resolve([]);
				}
				return api("/input-feeds?workflow_id=" + encodeURIComponent(id)).then(function(res) {
					const rows = asArray(res.feeds);
					const nextFeedId = rows.some(function(feed) {
						return feed.feed_id === selectedFeedId;
					}) ? selectedFeedId : rows[0] && rows[0].feed_id ? rows[0].feed_id : "";
					setInputFeeds(rows);
					setSelectedFeedId(nextFeedId);
					return nextFeedId ? loadInputFeedItems(nextFeedId).then(function() {
						return rows;
					}) : rows;
				}).catch(function() {
					setInputFeeds([]);
					setSelectedFeedId("");
					setInputFeedItems([]);
					return [];
				});
			}
			function openContinuousFeed() {
				const definition = selectedDefinition || {};
				const workflowId = workflowIdForDefinition(definition);
				const trigger = selectedInputTrigger(definition.spec);
				if (!workflowId) {
					setError("Deploy or select a workflow before opening an input feed.");
					return;
				}
				if (!trigger) {
					setError("This workflow has no manual trigger with intake.mode: continuous.");
					return;
				}
				setFeedBusy(true);
				setError("");
				api("/definitions/" + encodeURIComponent(workflowId) + "/input-feeds" + versionQuery(definition.version), {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ trigger_id: trigger && (trigger.id || trigger.name) || null })
				}).then(function(res) {
					const feed = res.feed || {};
					setSelectedFeedId(feed.feed_id || "");
					setStatus("Opened continuous input feed " + safeString(feed.feed_id));
					return loadInputFeeds(workflowId);
				}).catch(fail).finally(function() {
					setFeedBusy(false);
				});
			}
			function setSelectedFeedStatus(nextStatus) {
				const feedId = selectedFeedId || inputFeeds[0] && inputFeeds[0].feed_id;
				if (!feedId) return;
				setFeedBusy(true);
				api("/input-feeds/" + encodeURIComponent(feedId) + "/status", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ status: nextStatus })
				}).then(function() {
					setStatus("Set input feed " + safeString(feedId) + " to " + safeString(nextStatus));
					return loadInputFeeds();
				}).catch(fail).finally(function() {
					setFeedBusy(false);
				});
			}
			function manualTick() {
				setTicking(true);
				setError("");
				api("/tick", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ limit: 1 })
				}).then(function(res) {
					setStatus("Manual tick processed " + safeString(res.processed || 0) + " workflow(s).");
					return Promise.all([loadExecutions(), loadInputFeeds()]);
				}).catch(fail).finally(function() {
					setTicking(false);
				});
			}
			function updateInputFeedItem(item) {
				const itemId = item && item.item_id;
				const itemStatus = safeString(item && item.status);
				const feedId = selectedFeedId || inputFeeds[0] && inputFeeds[0].feed_id;
				if (!itemId || ["needs_input", "queued"].indexOf(itemStatus) === -1) return;
				const text = window.prompt("Update input item JSON", jsonBlock(item && item.input || {}));
				if (text === null) return;
				let input = {};
				try {
					input = JSON.parse(text || "{}");
				} catch (err) {
					fail(err);
					return;
				}
				setFeedBusy(true);
				setError("");
				api("/input-items/" + encodeURIComponent(itemId), {
					method: "PATCH",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ input })
				}).then(function(res) {
					const updated = res.item || {};
					setStatus("Updated input item " + safeString(updated.item_id || itemId) + " (" + safeString(updated.status) + ").");
					return Promise.all([loadInputFeeds(), loadInputFeedItems(feedId)]);
				}).catch(fail).finally(function() {
					setFeedBusy(false);
				});
			}
			function addItemToFeed(event) {
				if (event) event.preventDefault();
				const feedId = selectedFeedId || inputFeeds[0] && inputFeeds[0].feed_id;
				if (!feedId) {
					setError("Open a continuous input feed before adding items.");
					return;
				}
				let input = {};
				try {
					const spec = runInputSpec() || selectedDefinition && selectedDefinition.spec;
					input = showAdvancedFeedInputJson ? JSON.parse(feedInputText || "{}") : inputObjectForFields(inputFieldsForSpec(spec, selectedInputTrigger(spec)), feedInputValues);
				} catch (err) {
					fail(err);
					return;
				}
				setFeedBusy(true);
				setError("");
				api("/input-feeds/" + encodeURIComponent(feedId) + "/items", {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ input })
				}).then(function(res) {
					const item = res.item || {};
					setStatus("Added input item " + safeString(item.item_id) + " (" + safeString(item.status) + ").");
					setFeedInputValues({});
					setFeedInputText("{}");
					return loadInputFeeds();
				}).catch(fail).finally(function() {
					setFeedBusy(false);
				});
			}
			function selectNodeForInspector(item) {
				let node = item;
				if (item && item.specKind && item.spec && typeof item.spec === "object") node = item.specKind === "trigger" ? Object.assign({}, item.spec, {
					id: item.id,
					specKind: "trigger",
					type: "trigger",
					trigger_type: item.triggerType
				}) : Object.assign({}, item.spec, {
					id: item.id,
					specKind: "node",
					type: item.rendererType
				});
				setSelectedNode(node);
				setNodeJson(jsonBlock(node));
				setNodeMessage("");
				setAdvancedJsonOpen(false);
				setPromptAssistantOpen(false);
				const rawPrompt = node ? node.prompt : void 0;
				const assistantOutput = node && node.result_contract ? jsonBlock(node.result_contract) : "{\"summary\":\"string\",\"status\":\"string\"}";
				setCellId(node && (node.id || node.name) ? String(node.id || node.name) : "");
				setCellType(node && node.specKind === "trigger" ? String(node.trigger_type || node.type || "manual") : String(node && node.type || "pass"));
				setTriggerInputRows(node && node.specKind === "trigger" ? inputRowsFromTrigger(node) : []);
				setTriggerInputName("");
				setTriggerInputKind("text");
				setTriggerInputRequired(true);
				setTriggerInputDefault("");
				setTriggerInputMinLength("");
				setTriggerInputMaxLength("");
				setTriggerIntakeMode(node && node.specKind === "trigger" && node.intake && node.intake.mode === "continuous" ? "continuous" : "single");
				setTriggerDedupeKey(node && node.specKind === "trigger" && node.intake && node.intake.dedupe_key ? String(node.intake.dedupe_key) : "");
				setTriggerReadyPath(node && node.specKind === "trigger" ? readyPathFromTrigger(node) : "");
				setTriggerSchedule(node && (node.schedule || node.cron || node.expr) ? String(node.schedule || node.cron || node.expr) : "");
				setCellOutputText(node && node.output !== void 0 && node.output !== null ? typeof node.output === "string" ? node.output : jsonBlock(node.output) : "");
				setCellSeconds(node && node.seconds !== void 0 && node.seconds !== null ? String(node.seconds) : "60");
				setSwitchDefault(node && node.default ? String(node.default) : "");
				setSwitchCases(asArray(node && node.cases));
				setSwitchCaseName("");
				setSwitchCaseEquals("");
				setAgentProfile(node && node.profile ? String(node.profile) : "");
				setAgentProvider(providerValue(node));
				setAgentModel(modelValue(node));
				setAgentTitle(node && node.title ? String(node.title) : "");
				setPromptText(rawPrompt === null || rawPrompt === void 0 ? "" : typeof rawPrompt === "string" ? rawPrompt : jsonBlock(rawPrompt));
				setResultContractText(jsonBlock(node && node.result_contract || {}));
				setPromptAssistantObjective(node && node.title ? String(node.title) : "");
				setPromptAssistantContext("${ input }\n${ node.previous.output }");
				setPromptAssistantOutput(assistantOutput);
				setPromptAssistantConstraints("Return JSON only");
			}
			function loadDefinition(workflowId, version) {
				if (!workflowId) {
					setSelectedDefinition(null);
					setDraftSpec(null);
					setSelectedNode(null);
					return Promise.resolve(null);
				}
				const previousSelectionKey = definitionSelectionKey(selectedDefinition);
				return api("/definitions/" + encodeURIComponent(workflowId) + versionQuery(version)).then(function(res) {
					const definition = res.definition || null;
					const nextSelectionKey = definitionSelectionKey(definition);
					setSelectedDefinition(definition);
					setDraftSpec(definition && definition.spec ? definition.spec : null);
					if (nextSelectionKey !== previousSelectionKey) {
						setInputFieldValues({});
						setShowAdvancedInputJson(false);
						setRunInputText("{}");
					}
					setDraftResult(null);
					setRefineText("");
					if (definition && definition.spec) updateEditorText(specToEditorText(definition.spec));
					setSelectedNode(null);
					if (definition) {
						setRunWorkflowId(definition.workflow_id || definition.id || workflowId);
						loadInputFeeds(definition.workflow_id || definition.id || workflowId);
					} else loadInputFeeds("");
					return definition;
				});
			}
			function loadEvents(executionId) {
				if (!executionId) {
					setEvents([]);
					return Promise.resolve([]);
				}
				return api("/executions/" + encodeURIComponent(executionId) + "/events").then(function(res) {
					const rows = asArray(res.events);
					setEvents(rows);
					return rows;
				});
			}
			function loadNodeRuns(executionId) {
				if (!executionId) {
					setNodeRuns([]);
					return Promise.resolve([]);
				}
				return api("/executions/" + encodeURIComponent(executionId) + "/node-runs").then(function(res) {
					const rows = asArray(res.node_runs);
					setNodeRuns(rows);
					return rows;
				}).catch(function() {
					setNodeRuns([]);
					return [];
				});
			}
			function loadExecution(executionId) {
				if (!executionId) {
					setSelectedExecution(null);
					setEvents([]);
					setNodeRuns([]);
					return Promise.resolve(null);
				}
				return api("/executions/" + encodeURIComponent(executionId)).then(function(res) {
					const execution = res.execution || null;
					setSelectedExecution(execution);
					return Promise.all([loadEvents(executionId), loadNodeRuns(executionId)]).then(function() {
						return execution;
					});
				});
			}
			function loadDefinitions(preferId, preferVersion) {
				return SDK.fetchJSON(DEFINITIONS_API).then(function(res) {
					const rows = asArray(res.definitions);
					const currentId = workflowIdForDefinition(selectedDefinition);
					const currentVersion = selectedDefinition && selectedDefinition.version;
					const first = rows[0] || null;
					function hasDefinition(id) {
						return !!id && rows.some(function(definition) {
							return workflowIdForDefinition(definition) === String(id);
						});
					}
					const preferredId = hasDefinition(preferId) ? preferId : "";
					const selectedId = hasDefinition(currentId) ? currentId : "";
					const runId = hasDefinition(runWorkflowId) ? runWorkflowId : "";
					const nextId = preferredId || selectedId || runId || workflowIdForDefinition(first) || "";
					let nextVersion = preferredId ? preferVersion : selectedId ? currentVersion : void 0;
					if ((nextVersion === void 0 || nextVersion === null || nextVersion === "") && nextId) {
						const matches = rows.filter(function(definition) {
							return workflowIdForDefinition(definition) === String(nextId);
						});
						const match = matches[matches.length - 1];
						if (match) nextVersion = match.version;
					}
					setDefinitions(rows);
					if (nextId) return loadDefinition(nextId, nextVersion);
					setRunWorkflowId("");
					setSelectedDefinition(null);
					setDraftSpec(null);
					setSelectedNode(null);
					return null;
				});
			}
			function loadExecutions(preferId) {
				return api("/executions").then(function(res) {
					const rows = asArray(res.executions);
					const currentId = selectedExecution && selectedExecution.execution_id;
					function hasExecution(id) {
						return !!id && rows.some(function(execution) {
							return String(execution.execution_id || execution.id || "") === String(id);
						});
					}
					const nextId = (hasExecution(preferId) ? preferId : "") || (hasExecution(currentId) ? currentId : "") || rows[0] && rows[0].execution_id || "";
					setExecutions(rows);
					if (nextId) return loadExecution(nextId);
					setSelectedExecution(null);
					setEvents([]);
					setNodeRuns([]);
					return null;
				});
			}
			function loadWorkflowStatus() {
				return api("/status").then(setWorkflowStatus).catch(function() {
					setWorkflowStatus(null);
				});
			}
			function loadWorkflowCapabilities() {
				return api("/capabilities").then(setWorkflowCapabilities).catch(function() {
					setWorkflowCapabilities(null);
				});
			}
			function loadAgentRoutingOptions() {
				return api("/agent-routing-options").then(function(res) {
					setAgentRoutingOptions(res || {
						profiles: [],
						providers: [],
						default_provider: "",
						default_model: ""
					});
				}).catch(function() {
					setAgentRoutingOptions({
						profiles: [],
						providers: [],
						default_provider: "",
						default_model: ""
					});
				});
			}
			function refresh(preferExecutionId) {
				setLoading(true);
				setError("");
				loadWorkflowStatus();
				loadWorkflowCapabilities();
				loadAgentRoutingOptions();
				return Promise.all([loadDefinitions(), loadExecutions(preferExecutionId)]).catch(fail).finally(function() {
					setLoading(false);
				});
			}
			useEffect(function() {
				refresh(initialExecutionId);
			}, []);
			useEffect(function() {
				function handlePopState() {
					const next = modeForLocation(currentLocationShape());
					setWorkspaceMode(WORKSPACE_MODES.indexOf(next) === -1 ? "build" : next);
				}
				if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
					window.addEventListener("popstate", handlePopState);
					return function() {
						window.removeEventListener("popstate", handlePopState);
					};
				}
			}, []);
			useEffect(function() {
				const spec = activeSpec();
				const statuses = statusByNode(events);
				setFlowNodes(spec ? buildFlowNodes(spec, statuses, selectedNode, selectNodeForInspector, nodePositions) : []);
				setFlowEdges(spec ? buildFlowEdges(spec) : []);
			}, [
				draftSpec,
				editorText,
				events,
				selectedNode,
				nodePositions
			]);
			useEffect(function() {
				const key = graphItems(activeSpec() || {}).map(function(item) {
					return item.specKind + ":" + item.id;
				}).join("|");
				if (key && key !== membershipKeyRef.current && flowInstanceRef.current && typeof flowInstanceRef.current.fitView === "function") flowInstanceRef.current.fitView();
				membershipKeyRef.current = key;
			}, [
				draftSpec,
				editorText,
				flowNodes
			]);
			useEffect(function() {
				if (!error && !status) return void 0;
				const timer = setTimeout(clearBanners, error ? 12e3 : 6e3);
				return function() {
					clearTimeout(timer);
				};
			}, [error, status]);
			useEffect(function() {
				const executionId = selectedExecution && selectedExecution.execution_id;
				const executionStatus = selectedExecution && selectedExecution.status;
				if (!executionId || executionStatus !== "queued" && executionStatus !== "waiting") return void 0;
				const timer = setInterval(function() {
					loadExecution(executionId).catch(function() {});
					api("/executions").then(function(res) {
						setExecutions(asArray(res.executions));
					}).catch(function() {});
				}, 5e3);
				return function() {
					clearInterval(timer);
				};
			}, [selectedExecution && selectedExecution.execution_id, selectedExecution && selectedExecution.status]);
			function validateDefinition() {
				setValidating(true);
				setError("");
				api("/definitions/validate", {
					method: "POST",
					headers: { "Content-Type": "text/plain" },
					body: editorText
				}).then(function(res) {
					const definition = res.definition || {};
					setStatus("Validated " + safeString(definition.workflow_id || definition.id));
					setDraftSpec(definition.spec || null);
					if (definition.spec) updateEditorText(specToEditorText(definition.spec));
				}).catch(fail).finally(function() {
					setValidating(false);
				});
			}
			function deployDefinition() {
				setDeploying(true);
				setError("");
				api("/definitions/deploy", {
					method: "POST",
					headers: { "Content-Type": "text/plain" },
					body: editorText
				}).then(function(res) {
					const definition = res.definition || {};
					const id = definition.workflow_id || definition.id || "";
					const version = definition.version;
					setDraftSpec(definition.spec || null);
					setStatus("Deployed " + safeString(id) + ". Use Run to provide start input and launch an execution.");
					return loadDefinitions(id, version);
				}).catch(fail).finally(function() {
					setDeploying(false);
				});
			}
			function deleteWorkflow() {
				const workflowId = workflowIdForDefinition(selectedDefinition);
				if (!workflowId) return;
				const name = selectedDefinition && selectedDefinition.name ? selectedDefinition.name : workflowId;
				if (!confirm("Delete workflow \"" + safeString(name) + "\"? This deletes its versions, schedules, and execution history.")) return;
				setDeleting(true);
				setError("");
				api("/definitions/" + encodeURIComponent(workflowId), { method: "DELETE" }).then(function() {
					setSelectedDefinition(null);
					setDraftSpec(null);
					setDraftResult(null);
					setSelectedNode(null);
					setRunWorkflowId("");
					setSelectedExecution(null);
					setEvents([]);
					setNodeRuns([]);
					setNodePositions({});
					updateEditorText(specToEditorText(newWorkflowSpec("Untitled Workflow")));
					setStatus("Deleted workflow " + safeString(workflowId));
					return Promise.all([loadDefinitions("__deleted_workflow__"), loadExecutions("__deleted_execution__")]);
				}).catch(fail).finally(function() {
					setDeleting(false);
				});
			}
			function selectedRunVersion(workflowId) {
				const requested = String(workflowId || "");
				const selectedId = workflowIdForDefinition(selectedDefinition);
				if (selectedDefinition && selectedId === requested && selectedDefinition.version !== void 0 && selectedDefinition.version !== null) return selectedDefinition.version;
				const spec = activeSpec();
				if (spec && workflowIdForSpec(spec) === requested && versionForSpec(spec) !== null) return versionForSpec(spec);
				return null;
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
					if (showAdvancedInputJson) input = JSON.parse(runInputText || "{}");
					else input = inputObjectForFields(inputFieldsForSpec(runInputSpec()), inputFieldValues);
				} catch (err) {
					fail(err);
					return;
				}
				setRunning(true);
				setError("");
				const runVersion = selectedRunVersion(workflowId);
				api("/definitions/" + encodeURIComponent(workflowId) + "/run" + versionQuery(runVersion), {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ input })
				}).then(function(res) {
					const executionId = (res.execution || {}).execution_id;
					setRunPanelOpen(false);
					setStatus("Started execution " + safeString(executionId));
					return loadExecutions(executionId);
				}).catch(fail).finally(function() {
					setRunning(false);
				});
			}
			function cancelSelectedExecution() {
				const execution = selectedExecution || {};
				const executionId = execution.execution_id || execution.id || "";
				const executionStatus = safeString(execution.status);
				if (!executionId) {
					setStatus("Select an execution before cancelling it.");
					return;
				}
				if ([
					"succeeded",
					"failed",
					"cancelled",
					"blocked"
				].indexOf(executionStatus) !== -1) {
					setStatus("Cannot cancel terminal execution " + safeString(executionId));
					return;
				}
				setError("");
				api("/executions/" + encodeURIComponent(executionId) + "/cancel", { method: "POST" }).then(function(res) {
					setSelectedExecution(res.execution || execution);
					setStatus(res.cancelled ? "Cancelled execution " + safeString(executionId) : "Cannot cancel terminal execution " + safeString(executionId));
					return loadExecutions(executionId);
				}).catch(fail);
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
					body: JSON.stringify({ goal })
				}).then(function(res) {
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
				}).catch(fail).finally(function() {
					setDrafting(false);
				});
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
					body: JSON.stringify({
						spec,
						instruction
					})
				}).then(function(res) {
					const draft = res && res.draft || res || {};
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
				}).catch(fail).finally(function() {
					setRefining(false);
				});
			}
			function importDefinitionFile(event) {
				const file = event.target.files && event.target.files[0];
				if (!file) return;
				const reader = new FileReader();
				reader.onload = function() {
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
				reader.onerror = function() {
					setError("Could not import workflow file.");
				};
				reader.readAsText(file);
				event.target.value = "";
			}
			function exportYAML() {
				const spec = activeSpec();
				const base = spec && (spec.id || spec.workflow_id || spec.name) || runWorkflowId || "workflow";
				const blob = new Blob([editorText], { type: "text/yaml;charset=utf-8" });
				const url = URL.createObjectURL(blob);
				const anchor = document.createElement("a");
				anchor.href = url;
				anchor.download = classSafe(base) + ".yaml";
				anchor.click();
				setTimeout(function() {
					URL.revokeObjectURL(url);
				}, 0);
				setStatus("Export YAML downloaded " + anchor.download);
			}
			function copyYAML() {
				if (!navigator.clipboard) {
					setStatus("Clipboard API unavailable; use Export YAML instead.");
					return;
				}
				navigator.clipboard.writeText(editorText).then(function() {
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
				if (selectedNode.specKind === "trigger") spec.triggers = asArray(spec.triggers).map(function(trigger) {
					return (trigger.id || trigger.name) === selectedNode.id ? clean : trigger;
				});
				else if (Array.isArray(spec.nodes)) spec.nodes = spec.nodes.map(function(node) {
					return (node.id || node.name) === selectedNode.id ? clean : node;
				});
				else {
					spec.nodes = spec.nodes || {};
					if (nextId !== selectedNode.id) delete spec.nodes[selectedNode.id];
					const nodeValue = Object.assign({}, clean);
					delete nodeValue.id;
					spec.nodes[nextId] = nodeValue;
				}
				updateEditorText(JSON.stringify(spec, null, 2));
				setSelectedDefinition(Object.assign({}, selectedDefinition || {}, { spec }));
				setSelectedNode(Object.assign({}, nextNode, {
					id: nextId,
					specKind: selectedNode.specKind
				}));
				setNodeMessage("Applied node JSON to editor draft.");
			}
			function addTriggerInputFieldFromUi() {
				const name = workflowIdFromText(String(triggerInputName || "").trim()).replace(/-/g, "_");
				if (!name) {
					setNodeMessage("Enter an input field name before adding it.");
					return;
				}
				const row = {
					name,
					kind: SCALAR_INPUT_KINDS.indexOf(triggerInputKind) !== -1 ? triggerInputKind : "text",
					required: !!triggerInputRequired,
					defaultValue: triggerInputDefault,
					minLength: triggerInputMinLength,
					maxLength: triggerInputMaxLength
				};
				setTriggerInputRows(asArray(triggerInputRows).filter(function(existing) {
					return existing.name !== name;
				}).concat([row]));
				setTriggerInputName("");
				setTriggerInputDefault("");
				setTriggerInputMinLength("");
				setTriggerInputMaxLength("");
				setNodeMessage("Added input field " + safeString(name) + ". Apply to update the workflow draft.");
			}
			function removeTriggerInputField(name) {
				setTriggerInputRows(asArray(triggerInputRows).filter(function(row) {
					return row && row.name !== name;
				}));
			}
			function applyAgentCellForm() {
				if (!selectedNode) return;
				if (cellType !== "agent_task") {
					applyBasicCellForm();
					return;
				}
				const spec = activeSpec();
				if (!spec) {
					setNodeMessage("Validate the YAML draft before applying cell edits; no stale workflow was used.");
					return;
				}
				const nextId = cellId.trim() || selectedNode.id;
				const nextNode = Object.assign({}, selectedNode, {
					id: nextId,
					type: "agent_task",
					profile: agentProfile.trim(),
					title: agentTitle.trim() || nextId,
					prompt: promptText
				});
				const contractText = resultContractText.trim();
				if (contractText) {
					const contract = parseJsonObject(contractText);
					if (!contract) {
						setNodeMessage("Result contract JSON must be a JSON object.");
						return;
					}
					nextNode.result_contract = contract;
				} else delete nextNode.result_contract;
				const providerText = agentProvider.trim();
				const modelText = agentModel.trim();
				delete nextNode.provider_override;
				delete nextNode.model_override;
				if (providerText) nextNode.provider = providerText;
				else delete nextNode.provider;
				if (modelText) nextNode.model = modelText;
				else delete nextNode.model;
				const nextSpec = upsertSpecNode(spec, selectedNode.id, cleanedNodeForSpec(nextNode));
				updateEditorText(specToEditorText(nextSpec));
				setDraftSpec(nextSpec);
				setSelectedDefinition(Object.assign({}, selectedDefinition || {}, { spec: nextSpec }));
				setSelectedNode(Object.assign({}, nextNode, { id: nextId }));
				setNodeJson(jsonBlock(Object.assign({}, nextNode, { id: nextId })));
				setNodeMessage("Applied agent cell prompt to workflow draft.");
			}
			function applyBasicCellForm() {
				if (!selectedNode) return;
				const spec = activeSpec();
				if (!spec) {
					setNodeMessage("Validate or select a workflow before applying cell edits; no stale workflow was used.");
					return;
				}
				const nextId = cellId.trim() || selectedNode.id;
				const nextType = cellType.trim() || (selectedNode.specKind === "trigger" ? "manual" : "pass");
				const nextNode = Object.assign({}, selectedNode, {
					id: nextId,
					type: nextType
				});
				const titleText = agentTitle.trim();
				const promptValue = promptText.trim();
				const outputValue = cellOutputText.trim();
				if (titleText) nextNode.title = titleText;
				else delete nextNode.title;
				if (promptValue && selectedNode.specKind !== "trigger") nextNode.prompt = promptText;
				else delete nextNode.prompt;
				if (selectedNode.specKind !== "trigger") {
					if (nextType === "wait") nextNode.seconds = Math.max(0, parseInt(cellSeconds || "0", 10) || 0);
					else delete nextNode.seconds;
					if (nextType === "pass" || nextType === "fail") if (outputValue) nextNode.output = cellOutputText;
					else delete nextNode.output;
					else delete nextNode.output;
					if (nextType === "switch") {
						nextNode.cases = asArray(switchCases);
						if (switchDefault.trim()) nextNode.default = switchDefault.trim();
						else delete nextNode.default;
					} else {
						delete nextNode.cases;
						delete nextNode.default;
					}
				}
				delete nextNode.provider_override;
				delete nextNode.model_override;
				if (nextType === "agent_task") {
					const providerText = agentProvider.trim();
					const modelText = agentModel.trim();
					nextNode.profile = agentProfile.trim() || nextNode.profile || "default";
					if (!promptValue) nextNode.prompt = "Return JSON only matching the result contract.";
					if (!nextNode.result_contract || !Object.keys(nextNode.result_contract).length) nextNode.result_contract = {
						summary: "string",
						status: "string"
					};
					if (providerText) nextNode.provider = providerText;
					else delete nextNode.provider;
					if (modelText) nextNode.model = modelText;
					else delete nextNode.model;
				} else {
					delete nextNode.provider;
					delete nextNode.model;
				}
				if (selectedNode.specKind === "trigger") {
					const nextSpec = cloneSpec(spec);
					nextSpec.triggers = asArray(nextSpec.triggers).map(function(trigger) {
						if ((trigger.id || trigger.name) !== selectedNode.id) return trigger;
						const clean = cleanedNodeForSpec(Object.assign({}, trigger, {
							id: nextId,
							type: nextType
						}));
						if (titleText) clean.title = titleText;
						else delete clean.title;
						const schema = inputSchemaFromRows(triggerInputRows);
						if (Object.keys(schema).length) clean.input_schema = schema;
						else delete clean.input_schema;
						clean.intake = triggerIntakeFromForm(triggerIntakeMode, triggerDedupeKey, triggerReadyPath);
						if (nextType === "schedule") clean.schedule = triggerSchedule.trim() || trigger.schedule || trigger.cron || "0 9 * * *";
						else {
							delete clean.schedule;
							delete clean.cron;
							delete clean.expr;
						}
						return clean;
					});
					updateEditorText(specToEditorText(nextSpec));
					setDraftSpec(nextSpec);
					setSelectedDefinition(Object.assign({}, selectedDefinition || {}, { spec: nextSpec }));
					setSelectedNode(Object.assign({}, nextNode, {
						id: nextId,
						specKind: "trigger",
						trigger_type: nextType
					}));
					setNodeJson(jsonBlock(Object.assign({}, nextNode, {
						id: nextId,
						trigger_type: nextType
					})));
					setNodeMessage("Applied cell changes to workflow draft.");
					return;
				}
				const nextSpec = upsertSpecNode(spec, selectedNode.id, cleanedNodeForSpec(nextNode));
				updateEditorText(specToEditorText(nextSpec));
				setDraftSpec(nextSpec);
				setSelectedDefinition(Object.assign({}, selectedDefinition || {}, { spec: nextSpec }));
				setSelectedNode(nextNode);
				setNodeJson(jsonBlock(nextNode));
				setNodeMessage("Applied cell changes to workflow draft.");
			}
			function setActiveDraftSpec(nextSpec, message) {
				updateEditorText(specToEditorText(nextSpec));
				setDraftSpec(nextSpec);
				setSelectedDefinition(Object.assign({}, selectedDefinition || {}, { spec: nextSpec }));
				if (message) setStatus(message);
			}
			function startBlankWorkflow() {
				setActiveDraftSpec(newWorkflowSpec(newWorkflowName || goalText || "Untitled Workflow"), "Started blank workflow draft. No JSON/YAML required.");
				setSelectedNode(null);
				setNodeJson("");
				setNodeMessage("");
			}
			function addWorkflowCellOfType(type) {
				const safeType = type || "pass";
				const nextSpec = addSpecNodeAfter(activeSpec() || newWorkflowSpec(newWorkflowName || goalText || "Workflow Draft"), safeType, safeType, selectedNode && selectedNode.specKind !== "trigger" ? selectedNode.id : "");
				setActiveDraftSpec(nextSpec, "Added " + safeString(safeType) + " cell. Select it on the canvas to configure Properties.");
				const id = Object.keys(nextSpec.nodes || {}).slice(-1)[0];
				const node = findSpecNode(nextSpec, id);
				if (node) selectNodeForInspector(node);
			}
			function addTriggerOfType(type) {
				const safeType = type === "schedule" ? "schedule" : "manual";
				const nextSpec = addSpecTrigger(activeSpec() || newWorkflowSpec(newWorkflowName || goalText || "Workflow Draft"), safeType, safeType, newTriggerSchedule);
				setActiveDraftSpec(nextSpec, "Added " + safeString(safeType) + " trigger.");
				const trigger = asArray(nextSpec.triggers).slice(-1)[0];
				if (trigger) selectNodeForInspector(Object.assign({}, trigger, {
					id: trigger.id || trigger.name,
					specKind: "trigger",
					trigger_type: trigger.type
				}));
			}
			function addSwitchCaseFromUi() {
				if (!selectedNode) return;
				const spec = activeSpec();
				if (!spec) {
					setNodeMessage("Start or validate a workflow before adding switch cases.");
					return;
				}
				const nextSpec = addSwitchCaseToSpec(spec, selectedNode.id, switchCaseName || switchCaseEquals || "case", switchCasePath, switchCaseEquals || switchCaseName || "case");
				const nextNode = findSpecNode(nextSpec, selectedNode.id);
				setSwitchCases(asArray(nextNode && nextNode.cases));
				setActiveDraftSpec(nextSpec, "Added switch case. Connect cells from " + safeString(selectedNode.id) + "." + workflowIdFromText(switchCaseName || switchCaseEquals || "case") + " to route that branch.");
				if (nextNode) setSelectedNode(nextNode);
				setSwitchCaseName("");
				setSwitchCaseEquals("");
			}
			function addWorkflowCellAtPosition(type) {
				const safeType = type || "pass";
				const nextSpec = addSpecNodeAfter(activeSpec() || newWorkflowSpec(newWorkflowName || goalText || "Workflow Draft"), safeType, safeType, selectedNode && selectedNode.specKind !== "trigger" ? selectedNode.id : "");
				setActiveDraftSpec(nextSpec, "Added " + safeString(safeType) + " cell. Configure it in the inspector.");
				const id = Object.keys(nextSpec.nodes || {}).slice(-1)[0];
				const node = findSpecNode(nextSpec, id);
				if (node) selectNodeForInspector(node);
			}
			function deleteSelectedCell() {
				if (!selectedNode) return;
				const spec = activeSpec();
				if (!spec) {
					setNodeMessage("Start or validate a workflow before deleting a cell.");
					return;
				}
				const id = selectedNode.id || selectedNode.name;
				setActiveDraftSpec(selectedNode.specKind === "trigger" ? removeSpecTrigger(spec, id) : removeSpecNode(spec, id), "Deleted selected cell from workflow draft.");
				setSelectedNode(null);
				setNodeJson("");
				setNodeMessage("");
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
						provider: agentProvider,
						model: agentModel,
						cell_objective: promptAssistantObjective,
						available_context: promptAssistantContext.split(/\n+/).map(function(line) {
							return line.trim();
						}).filter(Boolean),
						expected_output: expectedOutput,
						constraints: promptAssistantConstraints.split(/\n+/).map(function(line) {
							return line.trim();
						}).filter(Boolean)
					})
				}).then(function(res) {
					setPromptText(res.prompt_text || "");
					setResultContractText(jsonBlock(res.result_contract || expectedOutput));
					setNodeMessage("Prompt assistant drafted a cell prompt. Review it, then Apply cell prompt.");
				}).catch(fail);
			}
			function renderValidationChecklist() {
				const spec = checklistSpec();
				if (!spec) return h(Card, { className: "hermes-workflows-panel hermes-workflows-validation-checklist" }, h("h2", null, "Validation checklist"), h("p", { className: "hermes-workflows-muted" }, "Select or validate a workflow to see the checklist."));
				const implementedTriggers = implementedTriggerTypesFromCapabilities(workflowCapabilities);
				const implementedNodes = implementedNodeTypesFromCapabilities(workflowCapabilities);
				const items = validationChecklist(spec, workflowCapabilities);
				return h(Card, { className: "hermes-workflows-panel hermes-workflows-validation-checklist" }, h("h2", null, "Validation checklist"), h("p", { className: "hermes-workflows-muted" }, "Checks implemented dashboard/dispatcher readiness, not every declared WorkflowSpec primitive."), h("ul", { className: "hermes-workflows-checklist" }, items.map(function(item) {
					return h("li", {
						key: item.label,
						className: "hermes-workflows-checklist-item " + (item.ok ? "is-ok" : "is-fail")
					}, h("span", {
						className: "hermes-workflows-checklist-mark",
						"aria-hidden": "true"
					}, item.ok ? "✓" : "!"), h("span", null, item.label));
				})), h("p", { className: "hermes-workflows-muted" }, "Implemented triggers today: " + implementedTriggers.join(", ") + "."), h("p", { className: "hermes-workflows-muted" }, "Implemented node types today: " + implementedNodes.join(", ") + "."));
			}
			function renderAdvancedYaml() {
				if (!showAdvancedYaml) return null;
				return h(Card, { className: "hermes-workflows-panel" }, h("h2", null, "Advanced YAML"), h("textarea", {
					className: "hermes-workflows-editor",
					value: editorText,
					onChange: function(event) {
						setDraftResult(null);
						setInputFieldValues({});
						setShowAdvancedInputJson(false);
						setRunInputText("{}");
						updateEditorText(event.target.value);
					}
				}), h("div", { className: "hermes-workflows-row" }, h("button", {
					type: "button",
					disabled: validating,
					onClick: validateDefinition
				}, validating ? "Validating…" : "Validate"), h("button", {
					type: "button",
					disabled: deploying,
					onClick: deployDefinition,
					className: "hermes-workflows-primary"
				}, deploying ? "Deploying…" : "Deploy"), h("label", { className: "hermes-workflows-file-button" }, "Import YAML", h("input", {
					type: "file",
					accept: ".yaml,.yml,.json,application/yaml,application/x-yaml,application/json",
					onChange: importDefinitionFile
				})), h("button", {
					type: "button",
					onClick: exportYAML
				}, "Export YAML"), h("button", {
					type: "button",
					onClick: copyYAML
				}, "Copy YAML")));
			}
			function renderExecutionStallWarning() {
				if (!selectedExecution) return null;
				const executionStatus = safeString(selectedExecution.status);
				if (executionStatus !== "queued" && executionStatus !== "waiting") return null;
				const dispatcher = workflowStatus && workflowStatus.dispatcher ? workflowStatus.dispatcher : null;
				if (!dispatcher || dispatcher.dispatch_in_gateway === true) return null;
				const warning = dispatcher.warning || "Set workflow.dispatch_in_gateway: true to let the gateway advance runs; fallback: hermes workflow tick.";
				return h("div", { className: "hermes-workflows-panel hermes-workflows-dispatcher-readiness is-warning" }, h("strong", null, "This execution will not advance automatically."), h("p", null, safeString(warning)));
			}
			function renderExecutionActions() {
				if (!selectedExecution) return null;
				const executionStatus = safeString(selectedExecution.status);
				const terminal = [
					"succeeded",
					"failed",
					"cancelled",
					"blocked"
				].indexOf(executionStatus) !== -1;
				return h("div", { className: "hermes-workflows-row" }, h("button", {
					type: "button",
					onClick: cancelSelectedExecution,
					disabled: terminal
				}, terminal ? "Cannot cancel terminal execution" : "Cancel Execution"));
			}
			function renderTimeline() {
				return h("div", { className: "hermes-workflows-timeline" }, h("p", { className: "hermes-workflows-privacy-note" }, PRIVACY_NOTE), renderExecutionStallWarning(), selectedExecution ? h("div", { className: "hermes-workflows-event" }, h("div", { className: "hermes-workflows-item-title" }, h("strong", null, safeString(selectedExecution.execution_id)), h("span", { className: "hermes-workflows-badge" }, safeString(selectedExecution.status))), h("div", { className: "hermes-workflows-meta" }, safeString(selectedExecution.workflow_id) + " · created " + safeString(selectedExecution.created_at)), h("pre", { className: "hermes-workflows-pre" }, jsonBlock(selectedExecution.input)), renderExecutionActions()) : h("p", { className: "hermes-workflows-muted" }, "Select an execution to inspect it."), renderNodeRuns(), events.length ? events.map(function(row) {
					return h("div", {
						key: row.id,
						className: "hermes-workflows-event"
					}, h("div", { className: "hermes-workflows-item-title" }, h("span", { className: "hermes-workflows-event-kind" }, safeString(row.kind)), h("span", { className: "hermes-workflows-meta" }, safeString(row.created_at))), row.node_run_id ? h("div", { className: "hermes-workflows-meta" }, "node run " + row.node_run_id) : null, h("pre", { className: "hermes-workflows-pre" }, jsonBlock(row.payload)));
				}) : h("p", { className: "hermes-workflows-muted" }, "No events recorded for this execution."));
			}
			function renderNodeRunPreview(label, value) {
				if (!hasPreviewValue(value)) return null;
				return h("div", { className: "hermes-workflows-node-run-preview" }, h("div", { className: "hermes-workflows-meta" }, label), h("pre", { className: "hermes-workflows-pre" }, previewJson(value)));
			}
			function renderNodeRuns() {
				if (!selectedExecution) return null;
				if (!nodeRuns.length) return h("p", { className: "hermes-workflows-muted" }, "No node runs recorded for this execution yet.");
				const groups = [];
				const byNode = {};
				nodeRuns.forEach(function(row, index) {
					const nodeId = safeString(row && row.node_id);
					if (!byNode[nodeId]) {
						byNode[nodeId] = [];
						groups.push({
							nodeId,
							rows: byNode[nodeId]
						});
					}
					byNode[nodeId].push({
						row: row || {},
						index
					});
				});
				return h("div", { className: "hermes-workflows-node-runs" }, h("h3", null, "Node runs"), groups.map(function(group) {
					return h("div", {
						key: group.nodeId,
						className: "hermes-workflows-node-run-group"
					}, group.rows.map(function(item) {
						const row = item.row;
						const status = safeString(row.status);
						const workerStatus = row.kanban_task_id && row.status === "waiting" ? "waiting on agent" : status;
						const payload = row.payload && typeof row.payload === "object" ? row.payload : {};
						const runProvider = row.provider || row.provider_override || payload.provider || payload.provider_override || "";
						const runModel = row.model || row.model_override || payload.model || payload.model_override || "";
						return h("div", {
							key: String(row.id || row.event_id || item.index),
							className: "hermes-workflows-node-run-card"
						}, h("div", { className: "hermes-workflows-item-title" }, h("strong", null, safeString(row.node_id)), h("span", { className: "hermes-workflows-badge" }, status)), row.kanban_task_id ? h("div", { className: "hermes-workflows-node-run-worker" }, h("div", null, "Linked worker task: " + safeString(row.kanban_task_id)), h("div", { className: "hermes-workflows-meta" }, "Worker status: " + workerStatus)) : null, runProvider || runModel ? h("div", { className: "hermes-workflows-node-run-routing" }, h("div", { className: "hermes-workflows-meta" }, "Provider / model: " + safeString(runProvider || "profile default") + " / " + safeString(runModel || "profile default"))) : null, renderNodeRunPreview("Output", row.output), renderNodeRunPreview("Error", row.error));
					}));
				}));
			}
			function renderCellList(spec) {
				const nodes = graphItems(spec);
				return h("section", {
					className: "hermes-workflows-cell-list",
					"aria-label": "Workflow cell list"
				}, h("h3", null, "Workflow cell list"), nodes.length ? nodes.map(function(item) {
					const id = item.id || item.spec.name || "node";
					const triggerType = item.specKind === "trigger" ? item.triggerType : item.rendererType;
					return h("button", {
						key: id,
						type: "button",
						className: "hermes-workflows-cell-list-item",
						"aria-label": "Edit cell " + safeString(id),
						onClick: function() {
							selectNodeForInspector(item);
						}
					}, h("span", null, safeString(id)), h("span", null, safeString(triggerType || "unknown")), h("span", null, triggerType === "agent_task" ? safeString(providerValue(item.spec) || "profile provider") : "—"), h("span", null, "Edit cell"));
				}) : h("p", { className: "hermes-workflows-muted" }, "No workflow cells available."));
			}
			function renderSimpleGraph(spec) {
				const items = graphItems(spec);
				const edges = edgeList(spec);
				return h("div", { className: "hermes-workflows-graph-fallback" }, renderCellList(spec), items.length ? h("div", { className: "hermes-workflows-node-grid" }, items.map(function(item) {
					const id = item.id || item.spec.name || "node";
					const triggerType = item.specKind === "trigger" ? item.triggerType : item.rendererType;
					return h("div", {
						key: id,
						className: "hermes-workflows-node-card",
						onClick: function() {
							selectNodeForInspector(item);
						}
					}, h("h3", null, safeString(id)), h("div", { className: "hermes-workflows-node-type" }, safeString(triggerType)), h("pre", { className: "hermes-workflows-pre" }, jsonBlock(item.spec)));
				})) : h("p", { className: "hermes-workflows-muted" }, "No nodes to render."), h("div", { className: "hermes-workflows-stack" }, h("h3", null, "Edges"), edges.length ? edges.map(function(edge) {
					const text = safeString(edge.from) + " → " + safeString(edge.to) + (edge.label ? " · " + edge.label : "");
					return h("div", {
						key: edge.id,
						className: "hermes-workflows-edge-card"
					}, text);
				}) : h("p", { className: "hermes-workflows-muted" }, "No edges defined.")));
			}
			function renderAdvancedNodeJson(spec) {
				return h("div", { className: "hermes-workflows-stack" }, h("h3", null, "Advanced JSON"), h("textarea", {
					className: "hermes-workflows-node-json",
					value: nodeJson,
					onChange: function(event) {
						setNodeJson(event.target.value);
					}
				}), h("div", { className: "hermes-workflows-row" }, h("button", {
					type: "button",
					onClick: applyNodeJson
				}, "Apply node JSON"), h("button", {
					type: "button",
					onClick: useJsonDraft,
					disabled: !spec
				}, "Use JSON draft")));
			}
			function renderPromptAssistant() {
				if (!promptAssistantOpen) return null;
				return h("div", { className: "hermes-workflows-assistant" }, h("h4", null, "Prompt assistant"), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Workflow goal"), h("textarea", {
					value: promptAssistantGoal,
					onChange: function(event) {
						setPromptAssistantGoal(event.target.value);
					}
				})), promptAssistantAdvanced ? h("div", { className: "hermes-workflows-stack" }, h("label", null, h("span", { className: "hermes-workflows-muted" }, "Cell objective"), h("textarea", {
					value: promptAssistantObjective,
					onChange: function(event) {
						setPromptAssistantObjective(event.target.value);
					}
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Context placeholders"), h("textarea", {
					value: promptAssistantContext,
					onChange: function(event) {
						setPromptAssistantContext(event.target.value);
					}
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Output contract JSON"), h("textarea", {
					value: promptAssistantOutput,
					onChange: function(event) {
						setPromptAssistantOutput(event.target.value);
					}
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Constraints"), h("textarea", {
					value: promptAssistantConstraints,
					onChange: function(event) {
						setPromptAssistantConstraints(event.target.value);
					}
				})), h("button", {
					type: "button",
					onClick: function() {
						setPromptAssistantAdvanced(false);
					},
					className: "hermes-workflows-link-btn"
				}, "Hide advanced fields")) : h("button", {
					type: "button",
					onClick: function() {
						setPromptAssistantAdvanced(true);
					},
					className: "hermes-workflows-link-btn"
				}, "Show advanced fields"), h("button", {
					type: "button",
					onClick: draftPromptWithAssistant,
					className: "hermes-workflows-primary"
				}, "Draft prompt"));
			}
			function renderAgentTaskInspector() {
				const profiles = profileRows(agentRoutingOptions);
				const providers = providerRows(agentRoutingOptions);
				const selectedProvider = providers.find(function(provider) {
					return String(provider.slug || provider.provider || "") === agentProvider;
				});
				const models = asArray(selectedProvider && selectedProvider.models);
				return h("div", { className: "hermes-workflows-stack" }, h("label", null, h("span", { className: "hermes-workflows-muted" }, "Assigned profile"), profiles.length ? h("select", {
					value: agentProfile,
					onChange: function(event) {
						setAgentProfile(event.target.value);
					}
				}, [h("option", {
					key: "",
					value: ""
				}, "Choose profile")].concat(profiles.map(function(profile) {
					const label = profile.name + (profile.provider || profile.model ? " · " + safeString(profile.provider) + " / " + safeString(profile.model) : "");
					return h("option", {
						key: profile.name,
						value: profile.name
					}, label);
				}))) : h("input", {
					value: agentProfile,
					onChange: function(event) {
						setAgentProfile(event.target.value);
					},
					placeholder: "reviewer"
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Provider override"), h("select", {
					value: agentProvider,
					onChange: function(event) {
						setAgentProvider(event.target.value);
						setAgentModel("");
					}
				}, [h("option", {
					key: "",
					value: ""
				}, "Use profile default provider")].concat(providers.map(function(provider) {
					const slug = String(provider.slug || provider.provider || "");
					const label = provider.label || slug;
					return h("option", {
						key: slug,
						value: slug
					}, safeString(label));
				})))), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Model override"), models.length ? h("select", {
					value: agentModel,
					onChange: function(event) {
						setAgentModel(event.target.value);
					}
				}, [h("option", {
					key: "",
					value: ""
				}, "Use profile default model")].concat(models.map(function(model) {
					return h("option", {
						key: String(model),
						value: String(model)
					}, safeString(model));
				}))) : h("input", {
					value: agentModel,
					onChange: function(event) {
						setAgentModel(event.target.value);
					},
					placeholder: agentProvider ? "Model name for selected provider" : "Use profile default model"
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Task title"), h("input", {
					value: agentTitle,
					onChange: function(event) {
						setAgentTitle(event.target.value);
					},
					placeholder: "Review change"
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Agent cell prompt"), h("textarea", {
					className: "hermes-workflows-prompt-editor",
					value: promptText,
					onChange: function(event) {
						setPromptText(event.target.value);
					},
					placeholder: "Tell the assigned profile exactly what to do. Use ${ input.foo } or ${ node.previous.output.bar } for workflow context."
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Result contract JSON"), h("textarea", {
					className: "hermes-workflows-contract-editor",
					value: resultContractText,
					onChange: function(event) {
						setResultContractText(event.target.value);
					}
				})), h("button", {
					type: "button",
					onClick: function() {
						setPromptAssistantOpen(!promptAssistantOpen);
					}
				}, promptAssistantOpen ? "Hide Prompt assistant" : "Prompt assistant"), renderPromptAssistant());
			}
			function renderTriggerInspector() {
				const rows = asArray(triggerInputRows);
				return h("div", { className: "hermes-workflows-stack" }, h("label", null, h("span", { className: "hermes-workflows-muted" }, "Trigger type"), h("input", {
					value: cellType,
					onChange: function(event) {
						setCellType(event.target.value);
					},
					placeholder: "manual",
					list: "workflow-trigger-type-options"
				})), cellType === "schedule" ? h("label", null, h("span", { className: "hermes-workflows-muted" }, "Schedule / cron"), h("input", {
					value: triggerSchedule,
					onChange: function(event) {
						setTriggerSchedule(event.target.value);
					},
					placeholder: "0 9 * * *"
				})) : null, h("div", {
					className: "hermes-workflows-trigger-editor",
					"aria-label": "Input schema"
				}, h("div", { className: "hermes-workflows-item-title" }, h("strong", null, "Input schema"), h("span", { className: "hermes-workflows-meta" }, "Advanced JSON remains available")), h("p", { className: "hermes-workflows-muted" }, INTAKE_SCOPE_NOTE), rows.length ? h("div", { className: "hermes-workflows-input-field-list" }, rows.map(function(row) {
					return h("div", {
						key: row.name,
						className: "hermes-workflows-input-field-row"
					}, h("span", null, safeString(row.name)), h("span", { className: "hermes-workflows-badge" }, safeString(row.kind)), h("span", { className: "hermes-workflows-meta" }, row.required ? "required" : "optional"), h("button", {
						type: "button",
						onClick: function() {
							removeTriggerInputField(row.name);
						}
					}, "Remove"));
				})) : h("p", { className: "hermes-workflows-muted" }, "No input fields yet. Add fields below, then Apply."), h("div", { className: "hermes-workflows-input-field-editor" }, h("label", null, h("span", { className: "hermes-workflows-muted" }, "Input field name"), h("input", {
					value: triggerInputName,
					onChange: function(event) {
						setTriggerInputName(event.target.value);
					},
					placeholder: "repo_path"
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Input field kind"), h("select", {
					value: triggerInputKind,
					onChange: function(event) {
						setTriggerInputKind(event.target.value);
					}
				}, SCALAR_INPUT_KINDS.map(function(kind) {
					return h("option", {
						key: kind,
						value: kind
					}, kind);
				}))), h("label", { className: "hermes-workflows-run-advanced-toggle" }, h("input", {
					type: "checkbox",
					checked: triggerInputRequired,
					onChange: function(event) {
						setTriggerInputRequired(event.target.checked);
					}
				}), h("span", null, "Required input")), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Default value"), h("input", {
					value: triggerInputDefault,
					onChange: function(event) {
						setTriggerInputDefault(event.target.value);
					},
					placeholder: "optional"
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Min length"), h("input", {
					type: "number",
					min: "0",
					value: triggerInputMinLength,
					onChange: function(event) {
						setTriggerInputMinLength(event.target.value);
					},
					placeholder: "0"
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Max length"), h("input", {
					type: "number",
					min: "0",
					value: triggerInputMaxLength,
					onChange: function(event) {
						setTriggerInputMaxLength(event.target.value);
					},
					placeholder: "optional"
				})), h("button", {
					type: "button",
					onClick: addTriggerInputFieldFromUi
				}, "Add input field"))), h("div", {
					className: "hermes-workflows-trigger-editor",
					"aria-label": "Intake mode"
				}, h("strong", null, "Intake mode"), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Mode"), h("select", {
					value: triggerIntakeMode,
					onChange: function(event) {
						setTriggerIntakeMode(event.target.value);
					}
				}, h("option", { value: "single" }, "single"), h("option", { value: "continuous" }, "continuous"))), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Dedupe key"), h("input", {
					value: triggerDedupeKey,
					onChange: function(event) {
						setTriggerDedupeKey(event.target.value);
					},
					placeholder: "$.input.repo_path"
				})), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Ready when field path"), h("input", {
					value: triggerReadyPath,
					onChange: function(event) {
						setTriggerReadyPath(event.target.value);
					},
					placeholder: "$.input.repo_path"
				}))));
			}
			function renderSwitchInspector() {
				return h("div", { className: "hermes-workflows-stack" }, h("label", null, h("span", { className: "hermes-workflows-muted" }, "Default target cell"), h("input", {
					value: switchDefault,
					onChange: function(event) {
						setSwitchDefault(event.target.value);
					},
					placeholder: "Optional target id; or connect from switch.default"
				})), h("div", { className: "hermes-workflows-meta" }, "Switch cases: " + (switchCases.length ? switchCases.map(function(item) {
					return item && item.name;
				}).filter(Boolean).join(", ") : "none yet")), h("div", { className: "hermes-workflows-row" }, h("input", {
					value: switchCaseName,
					onChange: function(event) {
						setSwitchCaseName(event.target.value);
					},
					placeholder: "Case name, e.g. approved"
				}), h("input", {
					value: switchCasePath,
					onChange: function(event) {
						setSwitchCasePath(event.target.value);
					},
					placeholder: "$.input.status"
				}), h("input", {
					value: switchCaseEquals,
					onChange: function(event) {
						setSwitchCaseEquals(event.target.value);
					},
					placeholder: "Equals value"
				}), h("button", {
					type: "button",
					onClick: addSwitchCaseFromUi
				}, "Add case")));
			}
			function renderWaitInspector() {
				return h("div", { className: "hermes-workflows-stack" }, h("label", null, h("span", { className: "hermes-workflows-muted" }, "Wait seconds"), h("input", {
					value: cellSeconds,
					onChange: function(event) {
						setCellSeconds(event.target.value);
					},
					placeholder: "60"
				})));
			}
			function renderPassFailInspector(kind) {
				return h("div", { className: "hermes-workflows-stack" }, h("label", null, h("span", { className: "hermes-workflows-muted" }, kind === "fail" ? "Failure message" : "Output text"), h("textarea", {
					className: "hermes-workflows-prompt-editor",
					value: cellOutputText,
					onChange: function(event) {
						setCellOutputText(event.target.value);
					},
					placeholder: kind === "fail" ? "Why this workflow should fail." : "Optional output text for this cell."
				})));
			}
			function renderMinimalInspector() {
				return h("div", { className: "hermes-workflows-stack" }, h("p", { className: "hermes-workflows-muted" }, "Connect incoming and outgoing edges on the canvas."), h("label", null, h("span", { className: "hermes-workflows-muted" }, "Notes"), h("textarea", {
					className: "hermes-workflows-prompt-editor",
					value: promptText,
					onChange: function(event) {
						setPromptText(event.target.value);
					},
					placeholder: "Optional notes for this cell."
				})));
			}
			function renderInspectorForType(spec) {
				if (!selectedNode) return null;
				var kind = selectedNode.specKind === "trigger" ? "trigger" : cellType || selectedNode.type || "pass";
				var header = h("div", { className: "hermes-workflows-inspector-header" }, h("strong", null, safeString(selectedNode.id)), h("span", { className: "hermes-workflows-type-badge" }, kind));
				var idField = h("label", null, h("span", { className: "hermes-workflows-muted" }, "ID"), h("input", {
					value: cellId,
					onChange: function(event) {
						setCellId(event.target.value);
					},
					placeholder: "cell-id"
				}));
				var typeField = selectedNode.specKind === "trigger" ? null : h("label", null, h("span", { className: "hermes-workflows-muted" }, "Cell type"), h("select", {
					value: cellType,
					"aria-label": "Change selected cell type",
					onChange: function(event) {
						setCellType(event.target.value);
					}
				}, [
					"pass",
					"switch",
					"agent_task",
					"wait",
					"parallel",
					"join",
					"fail"
				].map(function(type) {
					return h("option", {
						key: type,
						value: type
					}, type);
				})));
				var body;
				if (kind === "agent_task") body = renderAgentTaskInspector();
				else if (kind === "trigger") body = renderTriggerInspector();
				else if (kind === "switch") body = renderSwitchInspector();
				else if (kind === "wait") body = renderWaitInspector();
				else if (kind === "pass" || kind === "fail") body = renderPassFailInspector(kind);
				else body = renderMinimalInspector();
				return h("div", { className: "hermes-workflows-stack" }, header, idField, typeField, body, h("div", { className: "hermes-workflows-row" }, h("button", {
					type: "button",
					onClick: kind === "agent_task" ? applyAgentCellForm : applyBasicCellForm,
					className: "hermes-workflows-primary"
				}, "Apply"), h("button", {
					type: "button",
					onClick: deleteSelectedCell
				}, "Delete"), h("button", {
					type: "button",
					onClick: function() {
						setAdvancedJsonOpen(!advancedJsonOpen);
					}
				}, advancedJsonOpen ? "Hide JSON" : "Advanced JSON")), advancedJsonOpen ? renderAdvancedNodeJson(spec) : null, nodeMessage ? h("p", { className: "hermes-workflows-muted" }, nodeMessage) : null);
			}
			function renderInspector(spec) {
				return h("aside", { className: "hermes-workflows-inspector" }, h("h3", null, "Node inspector"), selectedNode ? renderInspectorForType(spec) : h("p", { className: "hermes-workflows-muted" }, "Choose a node from the palette, select it on the canvas, then configure it in Properties."));
			}
			function renderReactFlowGraph(spec) {
				if (!ReactFlow || !ReactFlowProvider) return renderSimpleGraph(spec);
				return h("div", { className: "hermes-workflows-flow-surface" }, h("div", {
					className: "hermes-workflows-canvas" + (isDragOver ? " hermes-workflows-canvas-drop-target" : ""),
					onDragOver: function(event) {
						event.preventDefault();
						if (!isDragOver) setIsDragOver(true);
					},
					onDragLeave: function() {
						setIsDragOver(false);
					},
					onDrop: function(event) {
						event.preventDefault();
						setIsDragOver(false);
						const type = event.dataTransfer && event.dataTransfer.getData("text/plain") || window.__HERMES_DRAG_NODE_TYPE || "";
						if (type === "manual" || type === "schedule") addTriggerOfType(type);
						else if (type) addWorkflowCellAtPosition(type);
						delete window.__HERMES_DRAG_NODE_TYPE;
					}
				}, h(ReactFlowProvider, null, h(ReactFlow, {
					nodes: flowNodes,
					edges: flowEdges,
					nodeTypes: NODE_TYPES,
					fitView: false,
					nodesDraggable: true,
					nodesConnectable: true,
					onInit: function(instance) {
						flowInstanceRef.current = instance;
					},
					onNodeClick: function(_, node) {
						if (node && node.data && node.data.node) selectNodeForInspector(node.data.node);
					},
					onNodesChange: applyNodeChanges ? function(changes) {
						setFlowNodes(applyNodeChanges(changes, flowNodes));
						changes.forEach(function(change) {
							if (change.type === "position" && change.position && change.dragging === false) setNodePositions(function(prev) {
								var next = Object.assign({}, prev);
								next[change.id] = change.position;
								return next;
							});
						});
					} : void 0,
					onEdgesChange: applyEdgeChanges ? function(changes) {
						setFlowEdges(applyEdgeChanges(changes, flowEdges));
					} : void 0,
					onConnect: addEdge ? function(connection) {
						const spec = activeSpec();
						if (isTriggerSource(spec, connection.source)) {
							setStatus("Triggers start workflows automatically; connect cells to other cells, not triggers.");
							return;
						}
						setFlowEdges(addEdge(Object.assign({
							label: "draft",
							markerEnd: { type: MarkerType.ArrowClosed }
						}, connection), flowEdges));
						if (spec && connection.source && connection.target) {
							const nextSpec = upsertSpecEdge(spec, connection.sourceHandle ? connection.source + "." + connection.sourceHandle : connection.source, connection.target);
							updateEditorText(specToEditorText(nextSpec));
							setDraftSpec(nextSpec);
							setStatus("Connection added to workflow draft.");
						} else setStatus("Draft connection added visually; validate/select a workflow to persist it.");
					} : void 0,
					onNodeContextMenu: function(event, node) {
						event.preventDefault();
						if (node && node.data && node.data.node) {
							selectNodeForInspector(node.data.node);
							setContextMenu({
								x: event.clientX,
								y: event.clientY,
								visible: true
							});
						}
					}
				}, Controls ? h(Controls, null) : null)), contextMenu.visible ? h(React.Fragment, null, h("div", {
					className: "hermes-workflows-context-menu-overlay",
					style: {
						position: "fixed",
						inset: 0,
						zIndex: 999
					},
					onClick: function() {
						setContextMenu({
							x: 0,
							y: 0,
							visible: false
						});
					}
				}), h("div", {
					className: "hermes-workflows-context-menu",
					style: {
						left: contextMenu.x + "px",
						top: contextMenu.y + "px",
						zIndex: 1e3
					},
					onClick: function() {
						setContextMenu({
							x: 0,
							y: 0,
							visible: false
						});
					}
				}, h("button", {
					type: "button",
					onClick: function() {
						deleteSelectedCell();
						setContextMenu({
							x: 0,
							y: 0,
							visible: false
						});
					}
				}, "Delete cell"))) : null));
			}
			function renderTopBar() {
				var spec = activeSpec();
				var wfName = spec ? safeString(spec.name || spec.id || spec.workflow_id) : "Untitled workflow";
				var hasDraft = !!spec;
				var persisted = !!(selectedDefinition && workflowIdForDefinition(selectedDefinition) && versionForDefinition(selectedDefinition));
				return h("div", { className: "hermes-workflows-topbar" }, h("div", { className: "hermes-workflows-topbar-left" }, h("span", { className: "hermes-workflows-topbar-name" }, wfName), h("span", { className: "hermes-workflows-topbar-status" }, persisted ? "v" + safeString(selectedDefinition.version) + " · enabled" : "draft")), renderWorkspaceTabsBar(), h("div", { className: "hermes-workflows-topbar-actions" }, h("button", {
					type: "button",
					disabled: validating || !hasDraft,
					onClick: validateDefinition
				}, validating ? "Validating…" : "Validate"), h("button", {
					type: "button",
					disabled: deploying || !hasDraft,
					onClick: deployDefinition,
					className: "hermes-workflows-primary"
				}, deploying ? "Deploying…" : "Deploy"), persisted ? h("button", {
					type: "button",
					disabled: deleting,
					onClick: deleteWorkflow,
					"aria-label": "Delete workflow"
				}, deleting ? "Deleting…" : "Delete") : null, persisted ? h("button", {
					type: "button",
					disabled: running,
					onClick: function() {
						setRunWorkflowId(workflowIdForDefinition(selectedDefinition));
						setRunPanelOpen(true);
					}
				}, running ? "Running…" : "Run") : null, h("button", {
					type: "button",
					disabled: loading,
					onClick: function() {
						refresh();
					}
				}, loading ? "Refreshing…" : "Refresh"), h("button", {
					type: "button",
					onClick: function() {
						setShowAdvancedYaml(!showAdvancedYaml);
					}
				}, showAdvancedYaml ? "Hide YAML" : "YAML")));
			}
			function renderWorkspaceTabsBar() {
				workflowIdFromLocation() || selectedDefinition && workflowIdForDefinition(selectedDefinition);
				const runDisabled = !selectedDefinition || !persistedRunCapable();
				return h("div", {
					role: "tablist",
					"aria-label": "Workflow workspace modes",
					className: "hermes-workflows-workspace-tabs"
				}, WORKSPACE_MODES.map(function(mode) {
					return h("button", {
						key: mode,
						type: "button",
						role: "tab",
						"data-workspace-mode": mode,
						"aria-selected": workspaceMode === mode ? "true" : "false",
						"aria-controls": "hermes-workflows-mode-" + mode,
						disabled: mode === "run" && runDisabled,
						...mode === "run" && runDisabled ? {
							"aria-disabled": "true",
							title: "Run is disabled until the workflow is published"
						} : {},
						className: "hermes-workflows-workspace-tab" + (workspaceMode === mode ? " is-active" : ""),
						onClick: function() {
							if (workspaceMode === mode) return;
							setWorkspaceMode(mode);
							pushMode(mode, mode === "run" ? { feed: selectedFeedId } : mode === "history" ? { execution: selectedExecution && selectedExecution.execution_id } : {});
						}
					}, mode === "build" ? "Build" : mode === "run" ? "Run" : "History");
				}));
			}
			function persistedRunCapable() {
				return !!(selectedDefinition && workflowIdForDefinition(selectedDefinition) && versionForDefinition(selectedDefinition));
			}
			function renderDiagnosticsPanel() {
				return h("section", { className: "hermes-workflows-diagnostics" }, h("button", {
					type: "button",
					className: "hermes-workflows-diagnostics-toggle",
					"aria-expanded": diagnosticsOpen ? "true" : "false",
					"aria-controls": "hermes-workflows-diagnostics-body",
					onClick: function() {
						setDiagnosticsOpen(!diagnosticsOpen);
					}
				}, diagnosticsOpen ? "Hide diagnostics" : "Show diagnostics"), diagnosticsOpen ? h("div", {
					id: "hermes-workflows-diagnostics-body",
					className: "hermes-workflows-diagnostics-body"
				}, h("p", { className: "hermes-workflows-muted" }, "Manual advance for queued workflows and dispatcher status."), renderExecutionStallWarning(), h("div", { className: "hermes-workflows-row" }, h("button", {
					type: "button",
					disabled: ticking,
					onClick: manualTick
				}, ticking ? "Ticking…" : "Manual Tick"), h("button", {
					type: "button",
					disabled: loading,
					onClick: function() {
						refresh();
					}
				}, loading ? "Refreshing…" : "Refresh"))) : null);
			}
			function renderSidebar() {
				var spec = activeSpec();
				var goalCollapsed = sidebarCollapsed.goal === void 0 ? !!spec : !!sidebarCollapsed.goal;
				var wfCollapsed = !!sidebarCollapsed.workflows;
				var execCollapsed = !!sidebarCollapsed.executions;
				function toggleSection(key) {
					var next = Object.assign({}, sidebarCollapsed);
					next[key] = !(key === "goal" ? goalCollapsed : !!sidebarCollapsed[key]);
					setSidebarCollapsed(next);
				}
				return h("aside", { className: "hermes-workflows-sidebar" }, h("div", { className: "hermes-workflows-sidebar-section hermes-workflows-goal-compact hermes-workflows-sidebar-collapsible" + (goalCollapsed ? " is-collapsed" : "") }, h("h3", { onClick: function() {
					toggleSection("goal");
				} }, spec ? "New workflow / prompt" : "New workflow"), h("p", {
					className: "hermes-workflows-muted",
					style: { fontSize: "0.78rem" }
				}, "Describe it or start from blank."), h("input", {
					value: newWorkflowName,
					onChange: function(event) {
						setNewWorkflowName(event.target.value);
					},
					placeholder: "Workflow name",
					style: {
						width: "100%",
						marginBottom: "0.35rem"
					}
				}), h("textarea", {
					"aria-label": "Describe workflow goal",
					value: goalText,
					onChange: function(event) {
						setGoalText(event.target.value);
					},
					placeholder: "Example: review code changes, run tests, then deploy if approved."
				}), h("div", {
					className: "hermes-workflows-row",
					style: { marginTop: "0.3rem" }
				}, h("button", {
					type: "button",
					disabled: drafting,
					onClick: draftFromGoal,
					className: "hermes-workflows-primary",
					style: { fontSize: "0.78rem" }
				}, drafting ? "Generating…" : "Generate From Prompt"), h("button", {
					type: "button",
					"aria-label": "Start from scratch",
					onClick: startBlankWorkflow,
					style: { fontSize: "0.78rem" }
				}, "Start From Scratch")), spec ? h("form", {
					className: "hermes-workflows-stack",
					style: { marginTop: "0.4rem" },
					onSubmit: refineWorkflow
				}, h("textarea", {
					value: refineText,
					onChange: function(event) {
						setRefineText(event.target.value);
					},
					placeholder: "Refine: add a step, change routing, etc.",
					"aria-label": "Refine workflow",
					style: {
						fontSize: "0.78rem",
						minHeight: "40px",
						resize: "vertical"
					}
				}), h("button", {
					type: "submit",
					disabled: refining,
					style: { fontSize: "0.78rem" }
				}, refining ? "Refining…" : "Refine")) : null), h("div", {
					className: "hermes-workflows-sidebar-section" + (wfCollapsed ? " hermes-workflows-sidebar-collapsible is-collapsed" : " hermes-workflows-sidebar-collapsible"),
					onClick: function() {
						toggleSection("workflows");
					}
				}, h("h3", null, "Workflows"), h("div", { className: "hermes-workflows-sidebar-list" }, definitions.length ? definitions.map(function(definition) {
					var id = definition.workflow_id || definition.id;
					var key = definitionSelectionKey(definition);
					return h("button", {
						key,
						type: "button",
						className: "hermes-workflows-sidebar-item" + (key === definitionSelectionKey(selectedDefinition) ? " is-selected" : ""),
						onClick: function(event) {
							event.stopPropagation();
							setError("");
							loadDefinition(definition.workflow_id, definition.version).catch(fail);
						}
					}, h("span", { className: "hermes-workflows-sidebar-item-title" }, safeString(definition.name || id)), h("span", { className: "hermes-workflows-sidebar-badge" + (definition.enabled ? " is-enabled" : "") }, definition.enabled ? "on" : "off"));
				}) : h("p", {
					className: "hermes-workflows-muted",
					style: { fontSize: "0.78rem" }
				}, "No workflows deployed."))), h("div", {
					className: "hermes-workflows-sidebar-section" + (execCollapsed ? " hermes-workflows-sidebar-collapsible is-collapsed" : " hermes-workflows-sidebar-collapsible"),
					onClick: function() {
						toggleSection("executions");
					}
				}, h("h3", null, "Executions"), h("div", { className: "hermes-workflows-sidebar-list" }, executions.length ? executions.slice(0, 20).map(function(execution) {
					var eid = safeString(execution.execution_id || execution.id);
					var execStatus = safeString(execution.status);
					var statusClass = execStatus === "succeeded" ? " is-succeeded" : execStatus === "failed" ? " is-failed" : "";
					return h("button", {
						key: eid,
						type: "button",
						className: "hermes-workflows-sidebar-item",
						onClick: function(event) {
							event.stopPropagation();
							loadExecution(eid).catch(fail);
						}
					}, h("span", { className: "hermes-workflows-sidebar-item-title" }, eid.slice(0, 16)), h("span", { className: "hermes-workflows-sidebar-badge" + statusClass }, execStatus));
				}) : h("p", {
					className: "hermes-workflows-muted",
					style: { fontSize: "0.78rem" }
				}, "No executions yet."))));
			}
			function renderBuilderToolbar(spec) {
				return h("div", { className: "hermes-workflows-builder-toolbar" }, h("div", { className: "hermes-workflows-palette-header" }, h("div", null, h("strong", null, "Nodes library"), h("p", { className: "hermes-workflows-muted" }, "Drag a node type onto the canvas, or click to add it.")), h("div", { className: "hermes-workflows-palette-help" }, "Connect cells by dragging between node handles on the canvas.")), h("div", { className: "hermes-workflows-node-palette" }, h("button", {
					type: "button",
					className: "hermes-workflows-palette-card",
					draggable: true,
					onDragStart: function(event) {
						event.dataTransfer.setData("text/plain", "manual");
						window.__HERMES_DRAG_NODE_TYPE = "manual";
					},
					"aria-label": "Add trigger",
					onClick: function() {
						addTriggerOfType("manual");
					}
				}, h("span", { className: "hermes-workflows-palette-icon" }, "⚡"), h("span", { className: "hermes-workflows-palette-title" }, "Manual trigger"), h("span", { className: "hermes-workflows-palette-desc" }, "Start the workflow")), [
					[
						"pass",
						"Pass",
						"Shape or summarize data."
					],
					[
						"agent_task",
						"Agent task",
						"Delegate work to a Hermes profile."
					],
					[
						"switch",
						"Switch",
						"Branch based on a value."
					],
					[
						"parallel",
						"Parallel",
						"Run independent branches."
					],
					[
						"join",
						"Join",
						"Wait for branches to complete."
					],
					[
						"wait",
						"Wait",
						"Pause before continuing."
					],
					[
						"fail",
						"Fail",
						"Stop with an error message."
					]
				].map(function(item) {
					return h("button", {
						key: item[0],
						type: "button",
						className: "hermes-workflows-palette-card",
						draggable: true,
						onDragStart: function(event) {
							event.dataTransfer.setData("text/plain", item[0]);
							window.__HERMES_DRAG_NODE_TYPE = item[0];
						},
						"aria-label": "Add workflow cell: " + item[0],
						onClick: function() {
							addWorkflowCellOfType(item[0]);
						}
					}, h("span", { className: "hermes-workflows-palette-icon" }, item[0] === "agent_task" ? "🤖" : item[0] === "switch" ? "◇" : item[0] === "parallel" ? "⇉" : item[0] === "join" ? "⇥" : item[0] === "wait" ? "⏱" : item[0] === "fail" ? "!" : "▣"), h("span", { className: "hermes-workflows-palette-title" }, item[1]), h("span", { className: "hermes-workflows-palette-desc" }, item[2]));
				})), h("datalist", { id: "workflow-cell-type-options" }, [
					"pass",
					"switch",
					"agent_task",
					"wait",
					"parallel",
					"join",
					"fail"
				].map(function(type) {
					return h("option", {
						key: type,
						value: type
					});
				})), h("datalist", { id: "workflow-trigger-type-options" }, ["manual", "schedule"].map(function(type) {
					return h("option", {
						key: type,
						value: type
					});
				})));
			}
			function renderFeedInputField(field) {
				var name = safeString(field && field.name);
				var kind = safeString(field && field.kind || "text");
				var label = safeString(field && field.label || name);
				var value = feedInputValues[name] === void 0 || feedInputValues[name] === null ? "" : feedInputValues[name];
				var disabled = !!(field && field.disabled);
				function updateValue(event) {
					var next = Object.assign({}, feedInputValues);
					next[name] = event.target.value;
					setFeedInputValues(next);
				}
				var hint = field && field.description ? h("span", { className: "hermes-workflows-muted" }, safeString(field.description)) : null;
				return h("label", {
					key: name,
					className: "hermes-workflows-run-field"
				}, h("span", null, label + (field && field.required ? " *" : "")), kind === "boolean" ? h("select", {
					value,
					disabled,
					onChange: updateValue
				}, h("option", { value: "" }, "Not set"), h("option", { value: "true" }, "true"), h("option", { value: "false" }, "false")) : kind === "json" || kind === "long_text" || kind === "prompt" || kind === "criteria" || kind === "document" ? h("textarea", {
					value,
					disabled,
					onChange: updateValue,
					placeholder: kind,
					rows: kind === "document" || kind === "prompt" || kind === "criteria" ? 5 : 3
				}) : h("input", {
					type: kind === "number" || kind === "integer" ? "number" : kind === "url" ? "url" : "text",
					step: kind === "integer" ? "1" : "any",
					value,
					disabled,
					onChange: updateValue,
					placeholder: kind
				}), hint);
			}
			function renderInputFeedPanel() {
				if (!selectedDefinition) return null;
				var spec = selectedDefinition.spec || null;
				var workflowId = workflowIdForDefinition(selectedDefinition);
				var fields = inputFieldsForSpec(spec, selectedInputTrigger(spec));
				var selectedFeed = inputFeeds.find(function(feed) {
					return feed.feed_id === selectedFeedId;
				}) || inputFeeds[0] || null;
				var feedId = selectedFeed && selectedFeed.feed_id;
				var feedOpen = selectedFeed && selectedFeed.status === "open";
				return h("div", { className: "hermes-workflows-input-feed-panel" }, h("div", { className: "hermes-workflows-item-title" }, h("strong", null, "Continuous input feed"), selectedFeed ? h("span", { className: "hermes-workflows-badge" }, safeString(selectedFeed.status)) : h("span", { className: "hermes-workflows-badge" }, "not open")), h("p", { className: "hermes-workflows-muted" }, "Open a feed, then add scalar repo paths, prompts, or criteria. Ready items launch normal executions as the dispatcher ticks."), h("p", { className: "hermes-workflows-muted" }, INTAKE_SCOPE_NOTE), h("div", { className: "hermes-workflows-row" }, h("button", {
					type: "button",
					disabled: feedBusy || !workflowId,
					onClick: openContinuousFeed,
					className: "hermes-workflows-primary"
				}, feedBusy ? "Opening…" : "Open Continuous Feed"), inputFeeds.length ? h("select", {
					value: selectedFeedId,
					onChange: function(event) {
						const id = event.target.value;
						setSelectedFeedId(id);
						loadInputFeedItems(id);
					}
				}, inputFeeds.map(function(feed) {
					return h("option", {
						key: feed.feed_id,
						value: feed.feed_id
					}, safeString(feed.status) + " · " + safeString(feed.feed_id).slice(0, 12));
				})) : null, feedId ? h("button", {
					type: "button",
					disabled: feedBusy,
					onClick: function() {
						setSelectedFeedStatus("open");
					}
				}, "Resume Feed") : null, feedId ? h("button", {
					type: "button",
					disabled: feedBusy,
					onClick: function() {
						setSelectedFeedStatus("paused");
					}
				}, "Pause Feed") : null, feedId ? h("button", {
					type: "button",
					disabled: feedBusy,
					onClick: function() {
						setSelectedFeedStatus("closed");
					}
				}, "Close Feed") : null, feedId ? h("button", {
					type: "button",
					disabled: feedBusy,
					onClick: function() {
						loadInputFeedItems(feedId);
					}
				}, "Refresh Feed Items") : null), feedId ? h("section", {
					className: "hermes-workflows-feed-items",
					"aria-label": "Input feed items"
				}, h("div", { className: "hermes-workflows-item-title" }, h("strong", null, "Feed items"), h("span", { className: "hermes-workflows-badge" }, String(inputFeedItems.length))), inputFeedItems.length ? inputFeedItems.map(function(item) {
					const itemStatus = safeString(item.status);
					const canUpdate = feedOpen && ["needs_input", "queued"].indexOf(itemStatus) !== -1;
					return h("div", {
						key: item.item_id,
						className: "hermes-workflows-feed-item"
					}, h("div", { className: "hermes-workflows-item-title" }, h("strong", null, safeString(item.item_id).slice(0, 18)), h("span", { className: "hermes-workflows-badge" }, itemStatus)), h("pre", { className: "hermes-workflows-pre" }, jsonBlock(item.input || {})), h("button", {
						type: "button",
						disabled: feedBusy || !canUpdate,
						onClick: function() {
							updateInputFeedItem(item);
						}
					}, canUpdate ? "Update Item From JSON" : "Item Not Editable"));
				}) : h("p", { className: "hermes-workflows-muted" }, "No feed items yet.")) : null, feedId ? h("form", {
					className: "hermes-workflows-stack",
					onSubmit: addItemToFeed
				}, !feedOpen ? h("p", { className: "hermes-workflows-muted" }, "This feed is " + safeString(selectedFeed.status) + "; resume it before adding items.") : null, fields.length && !showAdvancedFeedInputJson ? h("div", { className: "hermes-workflows-run-fields" }, fields.map(function(field) {
					return renderFeedInputField(Object.assign({}, field, { disabled: !feedOpen }));
				})) : null, h("label", { className: "hermes-workflows-run-advanced-toggle" }, h("input", {
					type: "checkbox",
					checked: showAdvancedFeedInputJson,
					disabled: !feedOpen,
					onChange: function(event) {
						setShowAdvancedFeedInputJson(event.target.checked);
					}
				}), h("span", null, "Use advanced JSON input")), showAdvancedFeedInputJson ? h("textarea", {
					value: feedInputText,
					disabled: !feedOpen,
					onChange: function(event) {
						setFeedInputText(event.target.value);
					},
					rows: 8,
					"aria-label": "Input feed item JSON"
				}) : null, h("button", {
					type: "submit",
					className: "hermes-workflows-primary",
					disabled: feedBusy || !feedOpen
				}, feedBusy ? "Adding…" : "Add Item To Feed")) : null);
			}
			function renderRunInputField(field) {
				var name = safeString(field && field.name);
				var kind = safeString(field && field.kind || "text");
				var value = inputFieldValues[name] === void 0 || inputFieldValues[name] === null ? "" : inputFieldValues[name];
				function updateValue(event) {
					var next = Object.assign({}, inputFieldValues);
					next[name] = event.target.value;
					setInputFieldValues(next);
				}
				return h("label", {
					key: name,
					className: "hermes-workflows-run-field"
				}, h("span", null, (field && field.label ? safeString(field.label) : name) + (field && field.required ? " *" : "")), kind === "boolean" ? h("select", {
					value,
					onChange: updateValue
				}, h("option", { value: "" }, "Not set"), h("option", { value: "true" }, "true"), h("option", { value: "false" }, "false")) : kind === "json" || kind === "long_text" || kind === "prompt" || kind === "criteria" || kind === "document" ? h("textarea", {
					value,
					onChange: updateValue,
					placeholder: kind === "document" ? "Paste document text" : kind,
					rows: kind === "document" || kind === "prompt" || kind === "criteria" ? 5 : 3
				}) : h("input", {
					type: kind === "number" || kind === "integer" ? "number" : kind === "url" ? "url" : "text",
					step: kind === "integer" ? "1" : "any",
					value,
					onChange: updateValue,
					placeholder: kind
				}));
			}
			function renderRunStartPanel() {
				if (!runPanelOpen) return null;
				var fields = inputFieldsForSpec(runInputSpec());
				return h("div", {
					className: "hermes-workflows-run-overlay",
					role: "dialog",
					"aria-modal": "true",
					"aria-label": "Start workflow run"
				}, h("form", {
					className: "hermes-workflows-run-panel",
					onSubmit: runWorkflow
				}, h("div", { className: "hermes-workflows-run-panel-header" }, h("div", null, h("h3", null, "Start Workflow Run"), h("p", { className: "hermes-workflows-muted" }, fields.length ? "Provide the manual trigger input for this execution." : "No start input fields are configured for this workflow. Running will use empty input.")), h("button", {
					type: "button",
					className: "hermes-workflows-link-button",
					onClick: function() {
						setRunPanelOpen(false);
					}
				}, "Close")), fields.length && !showAdvancedInputJson ? h("div", { className: "hermes-workflows-run-fields" }, fields.map(renderRunInputField)) : null, h("label", { className: "hermes-workflows-run-advanced-toggle" }, h("input", {
					type: "checkbox",
					checked: showAdvancedInputJson,
					onChange: function(event) {
						setShowAdvancedInputJson(event.target.checked);
					}
				}), h("span", null, "Use advanced JSON input")), showAdvancedInputJson ? h("textarea", {
					value: runInputText,
					onChange: function(event) {
						setRunInputText(event.target.value);
					},
					rows: 8,
					"aria-label": "Workflow input JSON"
				}) : null, h("div", { className: "hermes-workflows-run-actions" }, h("button", {
					type: "button",
					onClick: function() {
						setRunPanelOpen(false);
					}
				}, "Cancel"), h("button", {
					type: "submit",
					className: "hermes-workflows-primary",
					disabled: running
				}, running ? "Running…" : "Start Run"))));
			}
			function renderBottomPanel() {
				return h("div", { className: "hermes-workflows-bottom-panel" + (bottomCollapsed ? " is-collapsed" : "") }, h("div", { className: "hermes-workflows-bottom-tabs" }, h("button", {
					type: "button",
					className: "hermes-workflows-bottom-tab is-active",
					role: "tab",
					"aria-selected": "true"
				}, "Validation"), h("button", {
					type: "button",
					className: "hermes-workflows-bottom-toggle",
					onClick: function() {
						setBottomCollapsed(!bottomCollapsed);
					}
				}, bottomCollapsed ? "▴ Expand" : "▾ Collapse")), h("div", { className: "hermes-workflows-bottom-content" }, bottomCollapsed ? null : renderValidationChecklist()));
			}
			function renderBuildMode() {
				return h("section", {
					id: "hermes-workflows-mode-build",
					role: "tabpanel",
					"aria-label": "Build workflow",
					className: "hermes-workflows-build-mode"
				}, h("div", { className: "hermes-workflows-canvas-area" }, renderBuilderToolbar(activeSpec()), h("div", { className: "hermes-workflows-canvas-main" }, h("div", { className: "hermes-workflows-canvas-wrap" }, activeSpec() ? renderReactFlowGraph(activeSpec()) : h("div", {
					className: "hermes-workflows-muted",
					style: {
						padding: "2rem",
						textAlign: "center"
					}
				}, "No workflow loaded. Use the sidebar to draft a new workflow or select an existing one.")), activeSpec() ? h("div", { className: "hermes-workflows-inspector-panel" }, renderInspector(activeSpec())) : null)));
			}
			function renderRunMode() {
				return h("section", {
					id: "hermes-workflows-mode-run",
					role: "tabpanel",
					"aria-label": "Run workflow",
					className: "hermes-workflows-run-mode"
				}, renderInputFeedPanel(), renderDiagnosticsPanel());
			}
			function renderHistoryMode() {
				return h("section", {
					id: "hermes-workflows-mode-history",
					role: "tabpanel",
					"aria-label": "Workflow execution history",
					className: "hermes-workflows-history-mode"
				}, h("div", { className: "hermes-workflows-history-toolbar" }, h("button", {
					type: "button",
					disabled: loading,
					onClick: function() {
						refresh();
					}
				}, loading ? "Refreshing…" : "Refresh executions")), h("div", { className: "hermes-workflows-history-list" }, executions.length ? executions.slice(0, 50).map(function(execution) {
					var eid = safeString(execution.execution_id || execution.id);
					var execStatus = safeString(execution.status);
					var statusClass = execStatus === "succeeded" ? " is-succeeded" : execStatus === "failed" ? " is-failed" : "";
					return h("button", {
						key: eid,
						type: "button",
						"data-execution-id": eid,
						className: "hermes-workflows-history-row" + (selectedExecution && String(selectedExecution.execution_id || selectedExecution.id || "") === eid ? " is-selected" : ""),
						onClick: function() {
							loadExecution(eid).catch(fail);
							pushMode("history", { execution: eid });
						}
					}, h("span", { className: "hermes-workflows-history-row-id" }, eid.slice(0, 16)), h("span", { className: "hermes-workflows-history-row-status" + statusClass }, execStatus));
				}) : h("p", { className: "hermes-workflows-muted" }, "No executions yet.")), h("div", { className: "hermes-workflows-history-detail" }, renderTimeline()));
			}
			function renderActiveMode() {
				if (workspaceMode === "run") return renderRunMode();
				if (workspaceMode === "history") return renderHistoryMode();
				return renderBuildMode();
			}
			activeSpec();
			return h("div", { className: "hermes-workflows" }, h("div", { className: "hermes-workflows-app" }, renderTopBar(), error || status ? h("div", { className: "hermes-workflows-status-row" }, error ? h("div", {
				className: "hermes-workflows-banner is-error",
				role: "alert"
			}, h("span", null, error), h("button", {
				type: "button",
				className: "hermes-workflows-banner-close",
				"aria-label": "Dismiss alert",
				onClick: clearBanners
			}, "×")) : null, status ? h("div", {
				className: "hermes-workflows-banner",
				role: "status"
			}, h("span", null, status), h("button", {
				type: "button",
				className: "hermes-workflows-banner-close",
				"aria-label": "Dismiss alert",
				onClick: clearBanners
			}, "×")) : null) : null, h("div", { className: "hermes-workflows-body" }, renderSidebar(), renderActiveMode(), workspaceMode === "build" ? renderBottomPanel() : null), renderRunStartPanel(), showAdvancedYaml ? renderAdvancedYaml() : null));
		}
		REG.register("workflows", WorkflowsPage);
	})();
	//#endregion
})();
