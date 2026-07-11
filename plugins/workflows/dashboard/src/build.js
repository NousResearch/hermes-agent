// Build helpers: semantic diff for workflows, candidate state machine, and API
// helpers for durable draft/publish lifecycle.
//
// semanticWorkflowDiff() is a pure function. candidateReducer() returns plain
// state for synchronous actions and a Promise only when it needs the API.

var META_KEYS = ["id", "name", "version", "enabled"];

function sectionsEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function triggerKey(t) { return t && t.id ? t.id : ""; }
function nodeIdPairs(nodes) {
  var ids = Object.keys(nodes || {}).sort();
  return ids.map(function (id) { return { id: id, node: nodes[id] }; });
}

function triggerHasInput(trigger) {
  if (!trigger || !trigger.input_schema) return false;
  return Object.keys(trigger.input_schema).length > 0;
}

function edgesEqual(a, b) {
  function norm(list) {
    return (list || []).map(function (e) {
      return (e.from || "") + "->" + (e.to || "");
    }).sort();
  }
  return JSON.stringify(norm(a)) === JSON.stringify(norm(b));
}

export function semanticWorkflowDiff(before, after) {
  var sections = [];
  var b = before || {};
  var a = after || {};

  // metadata
  var metaItems = [];
  META_KEYS.forEach(function (key) {
    if (JSON.stringify(b[key]) !== JSON.stringify(a[key])) {
      metaItems.push(key + " changed");
    }
  });
  if (metaItems.length) {
    sections.push({ kind: "metadata", summary: "Metadata changed", items: metaItems });
  }

  // triggers
  var bTriggers = (b.triggers || []).map(triggerKey).sort();
  var aTriggers = (a.triggers || []).map(triggerKey).sort();
  var triggerItems = [];
  if (JSON.stringify(bTriggers) !== JSON.stringify(aTriggers)) {
    triggerItems.push("trigger list changed");
  }
  var allTriggerIds = {};
  (b.triggers || []).concat(a.triggers || []).forEach(function (t) { allTriggerIds[triggerKey(t)] = true; });
  Object.keys(allTriggerIds).forEach(function (tid) {
    var bt = (b.triggers || []).find(function (t) { return triggerKey(t) === tid; });
    var at = (a.triggers || []).find(function (t) { return triggerKey(t) === tid; });
    if (bt && at && !sectionsEqual(bt, at)) {
      if (triggerHasInput(at) && !triggerHasInput(bt)) {
        triggerItems.push(tid + " gained input schema");
      } else if (JSON.stringify(bt.input_schema) !== JSON.stringify(at.input_schema)) {
        triggerItems.push(tid + " input schema changed");
      }
    }
  });
  if (triggerItems.length) {
    sections.push({ kind: "triggers", summary: "Triggers changed", items: triggerItems });
  }

  // nodes
  var bNodes = nodeIdPairs(b.nodes);
  var aNodes = nodeIdPairs(a.nodes);
  var nodeItems = [];
  var bNodeIds = bNodes.map(function (p) { return p.id; });
  var aNodeIds = aNodes.map(function (p) { return p.id; });
  aNodeIds.forEach(function (id) {
    if (bNodeIds.indexOf(id) === -1) nodeItems.push("added node " + id);
  });
  bNodeIds.forEach(function (id) {
    if (aNodeIds.indexOf(id) === -1) nodeItems.push("removed node " + id);
  });
  var sharedIds = aNodeIds.filter(function (id) { return bNodeIds.indexOf(id) !== -1; });
  sharedIds.forEach(function (id) {
    var bn = (b.nodes || {})[id];
    var an = (a.nodes || {})[id];
    if (JSON.stringify(bn) !== JSON.stringify(an)) {
      if (bn.type !== an.type) {
        nodeItems.push(id + " type " + (bn.type || "?") + " -> " + (an.type || "?"));
      } else {
        nodeItems.push(id + " changed");
      }
    }
  });
  if (nodeItems.length) {
    sections.push({ kind: "nodes", summary: "Nodes changed", items: nodeItems });
  }

  // routing
  if (!edgesEqual(b.edges, a.edges)) {
    var routingItems = [];
    var bEdgeSet = {};
    (b.edges || []).forEach(function (e) { bEdgeSet[e.from + "->" + e.to] = true; });
    (a.edges || []).forEach(function (e) {
      var k = e.from + "->" + e.to;
      if (!bEdgeSet[k]) routingItems.push("added " + e.from + " -> " + e.to);
    });
    var aEdgeSet = {};
    (a.edges || []).forEach(function (e) { aEdgeSet[e.from + "->" + e.to] = true; });
    (b.edges || []).forEach(function (e) {
      var k = e.from + "->" + e.to;
      if (!aEdgeSet[k]) routingItems.push("removed " + e.from + " -> " + e.to);
    });
    sections.push({ kind: "routing", summary: "Routing changed", items: routingItems });
  }

  // runtime
  var runtimeItems = [];
  sharedIds.forEach(function (id) {
    var bn = (b.nodes || {})[id];
    var an = (a.nodes || {})[id];
    if (!bn || !an) return;
    var bRuntime = { retry: bn.retry, catch: bn.catch };
    var aRuntime = { retry: an.retry, catch: an.catch };
    if (JSON.stringify(bRuntime) !== JSON.stringify(aRuntime)) {
      if (an.retry && !bn.retry) runtimeItems.push(id + " added retry");
      if (an.catch && !bn.catch) runtimeItems.push(id + " added catch");
      if (bn.catch && !an.catch) runtimeItems.push(id + " removed catch");
      if (bn.retry && !an.retry) runtimeItems.push(id + " removed retry");
      if (an.retry && bn.retry && JSON.stringify(an.retry) !== JSON.stringify(bn.retry)) {
        runtimeItems.push(id + " retry policy changed");
      }
      if (an.catch && bn.catch && an.catch !== bn.catch) {
        runtimeItems.push(id + " catch target changed");
      }
    }
  });
  if (runtimeItems.length) {
    sections.push({ kind: "runtime", summary: "Runtime settings changed", items: runtimeItems });
  }

  return sections;
}

export function createCandidateState() {
  return {
    savedDraft: null,
    workingDraft: null,
    candidateDraft: null,
    candidateSource: null,
    undoStack: [],
    publishConflict: null,
  };
}

export function isDraftDirty(state) {
  return JSON.stringify(state.savedDraft) !== JSON.stringify(state.workingDraft);
}

export function diffForCandidate(state) {
  if (!state.candidateDraft) return [];
  return semanticWorkflowDiff(state.workingDraft || {}, state.candidateDraft);
}

export function pushUndo(state, value) {
  var next = state.undoStack.concat([value]);
  if (next.length > 20) next = next.slice(-20);
  return Object.assign({}, state, { undoStack: next });
}

export function acceptCandidate(state) {
  var withUndo = pushUndo(state, state.workingDraft);
  return Object.assign({}, withUndo, {
    workingDraft: state.candidateDraft,
    candidateDraft: null,
    candidateSource: null,
    publishConflict: null,
  });
}

export function rejectCandidate(state) {
  return Object.assign({}, state, {
    candidateDraft: null,
    candidateSource: null,
  });
}

export function recordPublishResult(state, result) {
  if (result && result.ok) {
    return Object.assign({}, state, {
      savedDraft: result.savedDraft || state.workingDraft,
      candidateDraft: null,
      candidateSource: null,
      publishConflict: null,
    });
  }
  return state;
}

export function applyServerDraft(state, result) {
  if (result && result.status === 409) {
    return Object.assign({}, state, {
      publishConflict: {
        code: result.code || "workflow_version_conflict",
        hint: result.hint || "Reload the latest version and review the draft again.",
      },
    });
  }
  return state;
}

// ponytail: returns plain state for sync actions, Promise only when API is needed.
// callers that pass the result directly as state (without await) work because
// sync branches return plain objects; only publish/accept/refresh return Promises.
export function candidateReducer(state, action, api) {
  var type = action.type;
  if (type === "set-working") {
    return Object.assign({}, state, { workingDraft: action.spec, publishConflict: null });
  }
  if (type === "set-saved") {
    return Object.assign({}, state, { savedDraft: action.draft });
  }
  if (type === "candidate") {
    return Object.assign({}, state, {
      candidateDraft: action.draft.spec,
      candidateSource: action.source,
      publishConflict: null,
    });
  }
  if (type === "accept") {
    var accepted = acceptCandidate(state);
    if (api && accepted.workingDraft) {
      var id = accepted.workingDraft.id || accepted.workingDraft.workflow_id || "";
      return api.putDraft(id, { spec: accepted.workingDraft, base_version: null }).then(function () {
        return Object.assign({}, accepted, { savedDraft: accepted.workingDraft });
      });
    }
    return accepted;
  }
  if (type === "reject") {
    return rejectCandidate(state);
  }
  if (type === "publish") {
    var publishId = (state.workingDraft && (state.workingDraft.id || state.workingDraft.workflow_id)) || "";
    // ponytail: wrap in Promise.resolve().then() so synchronous throws from the mock are caught.
    return Promise.resolve().then(function () {
      return api.publish(publishId, { expected_latest_version: action.expected_latest_version });
    }).then(function () {
      return recordPublishResult(state, { ok: true, savedDraft: state.workingDraft });
    }).catch(function (err) {
      if (err && err.status === 409) {
        if (err.code) throw err;
        return applyServerDraft(state, {
          status: 409,
          code: "workflow_version_conflict",
          hint: "Reload the latest version and review the draft again.",
        });
      }
      throw err;
    });
  }
  if (type === "refresh") {
    if (isDraftDirty(state)) return state;
    return Object.assign({}, state, {
      workingDraft: action.serverDraft,
      savedDraft: action.serverDraft,
    });
  }
  return state;
}

export function buildApiHelpers(api) {
  return {
    putDraft: function (workflowId, body) {
      return api(
        "/definitions/" + encodeURIComponent(workflowId) + "/draft",
        { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      );
    },
    getDraft: function (workflowId) {
      return api("/definitions/" + encodeURIComponent(workflowId) + "/draft", { method: "GET" });
    },
    deleteDraft: function (workflowId) {
      return api("/definitions/" + encodeURIComponent(workflowId) + "/draft", { method: "DELETE" });
    },
    publish: function (workflowId, body) {
      return api(
        "/definitions/" + encodeURIComponent(workflowId) + "/publish",
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
      );
    },
  };
}
