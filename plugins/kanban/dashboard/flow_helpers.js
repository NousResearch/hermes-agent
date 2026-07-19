/* Shared Flow-view helpers. Loaded before the Kanban dashboard entry and directly importable in Node tests. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.HermesKanbanFlowHelpers = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  const GRAPH_NODE_W = 260;
  const GRAPH_NODE_H = 116;
  const GRAPH_PRESETS = {
    "balanced-horizontal": { direction: "horizontal", rankGap: 88, laneGap: 34 },
    "balanced-vertical": { direction: "vertical", rankGap: 72, laneGap: 34 },
    compact: { direction: "horizontal", rankGap: 52, laneGap: 18 },
  };

  function mean(values) {
    return values.length
      ? values.reduce(function (sum, value) { return sum + value; }, 0) / values.length
      : 0;
  }

  function stableTaskCompare(byId, a, b) {
    const at = byId.get(a);
    const bt = byId.get(b);
    return (bt.priority || 0) - (at.priority || 0) || a.localeCompare(b);
  }

  function resolveCrossAxis(items, targetById, itemSize, gap) {
    const ordered = items.slice().sort(function (a, b) {
      return (targetById.get(a) - targetById.get(b)) || a.localeCompare(b);
    });
    const placed = new Map();
    let edge = 0;
    ordered.forEach(function (id) {
      const desired = targetById.get(id) - itemSize / 2;
      const position = Math.max(edge, desired);
      placed.set(id, position);
      edge = position + itemSize + gap;
    });
    if (ordered.length > 0) {
      const desiredMean = mean(ordered.map(function (id) { return targetById.get(id); }));
      const actualMean = mean(ordered.map(function (id) { return placed.get(id) + itemSize / 2; }));
      const shift = Math.min(0, desiredMean - actualMean);
      if (shift < 0) {
        ordered.forEach(function (id) { placed.set(id, placed.get(id) + shift); });
      }
      const minimum = Math.min(...ordered.map(function (id) { return placed.get(id); }));
      if (minimum < 0) {
        ordered.forEach(function (id) { placed.set(id, placed.get(id) - minimum); });
      }
    }
    return placed;
  }

  function reconcileCrossAxis(laneRanks, lanes, parents, children, crossPosition, itemSize, gap) {
    const constraints = [];
    laneRanks.forEach(function (laneRank) {
      lanes.get(laneRank).forEach(function (id) {
        const parentIds = parents.get(id) || [];
        const childIds = children.get(id) || [];
        if (parentIds.length > 1) constraints.push({ id, neighborIds: parentIds });
        if (childIds.length > 1) constraints.push({ id, neighborIds: childIds });
      });
    });
    if (constraints.length === 0) return;

    const centers = new Map();
    crossPosition.forEach(function (position, id) {
      centers.set(id, position + itemSize / 2);
    });
    const laneOrders = laneRanks.map(function (laneRank) {
      return lanes.get(laneRank).slice().sort(function (a, b) {
        return (centers.get(a) - centers.get(b)) || a.localeCompare(b);
      });
    });
    const separation = itemSize + gap;

    for (let iteration = 0; iteration < 64; iteration += 1) {
      constraints.forEach(function (constraint) {
        const residual = centers.get(constraint.id) - mean(constraint.neighborIds.map(function (id) {
          return centers.get(id);
        }));
        const neighborCount = constraint.neighborIds.length;
        centers.set(
          constraint.id,
          centers.get(constraint.id) - residual * neighborCount / (neighborCount + 1),
        );
        constraint.neighborIds.forEach(function (id) {
          centers.set(id, centers.get(id) + residual / (neighborCount + 1));
        });
      });
      laneOrders.forEach(function (ordered) {
        for (let index = 1; index < ordered.length; index += 1) {
          const before = ordered[index - 1];
          const after = ordered[index];
          const overlap = separation - (centers.get(after) - centers.get(before));
          if (overlap > 0) {
            centers.set(before, centers.get(before) - overlap / 2);
            centers.set(after, centers.get(after) + overlap / 2);
          }
        }
      });
    }

    laneOrders.forEach(function (ordered) {
      const centerBefore = mean(ordered.map(function (id) { return centers.get(id); }));
      const finalSeparation = separation + 1e-9;
      for (let index = 1; index < ordered.length; index += 1) {
        const before = ordered[index - 1];
        const after = ordered[index];
        centers.set(after, Math.max(centers.get(after), centers.get(before) + finalSeparation));
      }
      const centerAfter = mean(ordered.map(function (id) { return centers.get(id); }));
      const recenter = centerBefore - centerAfter;
      ordered.forEach(function (id) {
        centers.set(id, centers.get(id) + recenter);
      });
    });

    let minimum = Infinity;
    centers.forEach(function (value) { minimum = Math.min(minimum, value - itemSize / 2); });
    const shift = minimum < 0 ? -minimum : 0;
    centers.forEach(function (value, id) {
      crossPosition.set(id, value - itemSize / 2 + shift);
    });
  }

  const GRAPH_STATUS_LABELS = {
    triage: "Triage",
    todo: "Todo",
    scheduled: "Scheduled",
    ready: "Ready",
    running: "In progress",
    blocked: "Blocked",
    review: "Review",
    done: "Done",
    archived: "Archived",
  };

  function graphCountLabel(count, singular, plural) {
    return count + " " + (count === 1 ? singular : (plural || singular + "s"));
  }

  function beginLayoutSettingsRequest(currentBoardRef, requestIdRef) {
    const requestBoard = currentBoardRef.current;
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;
    return { requestBoard, requestId };
  }

  function isCurrentLayoutSettingsRequest(currentBoardRef, requestIdRef, requestBoard, requestId) {
    return currentBoardRef.current === requestBoard && requestIdRef.current === requestId;
  }

  function invalidateLayoutSettingsRequests(requestIdRef) {
    requestIdRef.current += 1;
  }

  function flattenGraphTasks(board) {
    const out = [];
    for (const column of (board && board.columns) || []) {
      for (const task of column.tasks || []) out.push(task);
    }
    return out;
  }

  function connectedGraphComponents(tasks, links) {
    const ids = new Set(tasks.map(function (task) { return task.id; }));
    const neighbors = new Map();
    for (const id of ids) neighbors.set(id, []);
    for (const link of links) {
      if (!ids.has(link.parent_id) || !ids.has(link.child_id)) continue;
      neighbors.get(link.parent_id).push(link.child_id);
      neighbors.get(link.child_id).push(link.parent_id);
    }
    const components = [];
    const seen = new Set();
    for (const task of tasks) {
      if (seen.has(task.id)) continue;
      const queue = [task.id];
      const component = [];
      seen.add(task.id);
      while (queue.length > 0) {
        const id = queue.shift();
        component.push(id);
        for (const next of neighbors.get(id) || []) {
          if (!seen.has(next)) { seen.add(next); queue.push(next); }
        }
      }
      components.push(component);
    }
    const linked = components.filter(function (component) {
      return component.length > 1 || links.some(function (link) {
        return link.parent_id === component[0] || link.child_id === component[0];
      });
    });
    const unlinked = components.filter(function (component) {
      return component.length === 1 && !links.some(function (link) {
        return link.parent_id === component[0] || link.child_id === component[0];
      });
    }).flat();
    if (unlinked.length > 0) linked.push(unlinked);
    return linked;
  }

  function graphComponentLabel(componentTasks, componentIndex, isUnlinked) {
    if (isUnlinked) return "Unlinked tasks";
    const tenants = Array.from(new Set(componentTasks.map(function (task) {
      return task.tenant || "";
    }).filter(Boolean)));
    if (tenants.length === 1) return tenants[0];
    const root = componentTasks.find(function (task) {
      return !task.parent_task_id;
    });
    if (root && root.title) return root.title;
    return "Workflow " + (componentIndex + 1);
  }

  function buildTaskGraphLayout(board, matchingBoard, preset) {
    const config = GRAPH_PRESETS[preset] || GRAPH_PRESETS["balanced-horizontal"];
    const tasks = flattenGraphTasks(board);
    const byId = new Map(tasks.map(function (task) { return [task.id, task]; }));
    const boardLinks = board && Array.isArray(board.links) ? board.links : [];
    const links = boardLinks.filter(function (link) {
      return link !== null && typeof link === "object"
        && Object.prototype.hasOwnProperty.call(link, "parent_id")
        && Object.prototype.hasOwnProperty.call(link, "child_id")
        && byId.has(link.parent_id) && byId.has(link.child_id);
    });
    const matchingIds = new Set(flattenGraphTasks(matchingBoard).map(function (task) {
      return task.id;
    }));
    const components = connectedGraphComponents(tasks, links);
    const nodes = [];
    const islands = [];
    let top = 24;
    let graphWidth = 720;

    components.forEach(function (componentIds, componentIndex) {
      const componentSet = new Set(componentIds);
      const componentTasks = componentIds.map(function (id) { return byId.get(id); }).filter(Boolean);
      const componentLinks = links.filter(function (link) {
        return componentSet.has(link.parent_id) && componentSet.has(link.child_id);
      });
      const isUnlinked = componentLinks.length === 0;
      const incoming = new Map(componentIds.map(function (id) { return [id, 0]; }));
      const parents = new Map(componentIds.map(function (id) { return [id, []]; }));
      const children = new Map(componentIds.map(function (id) { return [id, []]; }));
      for (const link of componentLinks) {
        incoming.set(link.child_id, (incoming.get(link.child_id) || 0) + 1);
        parents.get(link.child_id).push(link.parent_id);
        children.get(link.parent_id).push(link.child_id);
      }
      componentIds.forEach(function (id) {
        parents.get(id).sort(function (a, b) { return stableTaskCompare(byId, a, b); });
        children.get(id).sort(function (a, b) { return stableTaskCompare(byId, a, b); });
      });

      const rank = new Map();
      const queue = componentIds.filter(function (id) { return incoming.get(id) === 0; })
        .sort(function (a, b) { return stableTaskCompare(byId, a, b); });
      for (const id of queue) rank.set(id, 0);
      let cursor = 0;
      while (cursor < queue.length) {
        const id = queue[cursor++];
        const nextRank = (rank.get(id) || 0) + 1;
        for (const childId of children.get(id) || []) {
          rank.set(childId, Math.max(rank.get(childId) || 0, nextRank));
          incoming.set(childId, incoming.get(childId) - 1);
          if (incoming.get(childId) === 0) queue.push(childId);
        }
      }
      let maxRank = 0;
      for (const value of rank.values()) maxRank = Math.max(maxRank, value);
      for (const id of componentIds) {
        if (!rank.has(id)) rank.set(id, maxRank + 1);
      }
      maxRank = 0;
      for (const value of rank.values()) maxRank = Math.max(maxRank, value);

      const lanes = new Map();
      for (const id of componentIds) {
        const value = isUnlinked ? 0 : rank.get(id);
        if (!lanes.has(value)) lanes.set(value, []);
        lanes.get(value).push(id);
      }
      const laneRanks = Array.from(lanes.keys()).sort(function (a, b) { return a - b; });
      laneRanks.forEach(function (laneRank) {
        lanes.get(laneRank).sort(function (a, b) { return stableTaskCompare(byId, a, b); });
      });

      const crossNodeSize = config.direction === "horizontal" ? GRAPH_NODE_H : GRAPH_NODE_W;
      const crossPosition = new Map();
      laneRanks.forEach(function (laneRank) {
        lanes.get(laneRank).forEach(function (id, laneIndex) {
          crossPosition.set(id, laneIndex * (crossNodeSize + config.laneGap));
        });
      });

      function sweepLane(laneIds, neighborsById) {
        const targetById = new Map();
        laneIds.forEach(function (id) {
          const neighborIds = neighborsById.get(id) || [];
          const desiredCenter = neighborIds.length
            ? mean(neighborIds.map(function (neighborId) {
              return crossPosition.get(neighborId) + crossNodeSize / 2;
            }))
            : crossPosition.get(id) + crossNodeSize / 2;
          targetById.set(id, desiredCenter);
        });
        const placed = resolveCrossAxis(laneIds, targetById, crossNodeSize, config.laneGap);
        laneIds.forEach(function (id) { crossPosition.set(id, placed.get(id)); });
      }

      for (let iteration = 0; iteration < 4; iteration += 1) {
        laneRanks.forEach(function (laneRank) {
          sweepLane(lanes.get(laneRank), parents);
        });
        laneRanks.slice().reverse().forEach(function (laneRank) {
          sweepLane(lanes.get(laneRank), children);
        });
      }
      reconcileCrossAxis(
        laneRanks,
        lanes,
        parents,
        children,
        crossPosition,
        crossNodeSize,
        config.laneGap,
      );

      let contentWidth = 0;
      let contentHeight = 0;
      const componentNodes = [];
      laneRanks.forEach(function (laneRank) {
        lanes.get(laneRank).forEach(function (id) {
          const localX = config.direction === "horizontal"
            ? laneRank * (GRAPH_NODE_W + config.rankGap)
            : crossPosition.get(id);
          const localY = config.direction === "horizontal"
            ? crossPosition.get(id)
            : laneRank * (GRAPH_NODE_H + config.rankGap);
          contentWidth = Math.max(contentWidth, localX + GRAPH_NODE_W);
          contentHeight = Math.max(contentHeight, localY + GRAPH_NODE_H);
          componentNodes.push({ id, localX, localY });
        });
      });

      const islandWidth = contentWidth + 56;
      const islandHeight = contentHeight + 82;
      const label = graphComponentLabel(componentTasks, componentIndex, isUnlinked);
      const archiveIds = Array.from(new Set(componentTasks
        .map(function (task) { return task.workflow_archive_id || null; })
        .filter(Boolean)));
      islands.push({
        id: "island-" + componentIndex,
        label,
        count: componentIds.length,
        active: componentTasks.filter(function (task) { return task.status !== "done"; }).length,
        x: 20,
        y: top,
        width: islandWidth,
        height: islandHeight,
        isUnlinked,
        archiveId: archiveIds.length === 1 ? archiveIds[0] : null,
        taskIds: componentIds.slice(),
        seedTaskId: componentIds[0],
      });
      componentNodes.forEach(function (componentNode) {
        nodes.push({
          id: componentNode.id,
          task: byId.get(componentNode.id),
          x: 48 + componentNode.localX,
          y: top + 58 + componentNode.localY,
          width: GRAPH_NODE_W,
          height: GRAPH_NODE_H,
          componentIndex,
          dimmed: !!matchingBoard.__highlightMatches && !matchingIds.has(componentNode.id),
        });
      });
      top += islandHeight + 28;
      graphWidth = Math.max(graphWidth, islandWidth + 40);
    });

    const positioned = new Map(nodes.map(function (node) { return [node.id, node]; }));
    const edges = links.map(function (link) {
      const source = positioned.get(link.parent_id);
      const target = positioned.get(link.child_id);
      if (!source || !target) return null;
      if (config.direction === "vertical") {
        const sx = source.x + source.width / 2;
        const sy = source.y + source.height;
        const tx = target.x + target.width / 2;
        const ty = target.y;
        const middle = sy + Math.max(36, (ty - sy) / 2);
        return {
          id: link.parent_id + "->" + link.child_id,
          source: source.task,
          target: target.task,
          d: "M " + sx + " " + sy + " C " + sx + " " + middle + ", " + tx + " " + middle + ", " + tx + " " + ty,
        };
      }
      const sx = source.x + source.width;
      const sy = source.y + source.height / 2;
      const tx = target.x;
      const ty = target.y + target.height / 2;
      const middle = sx + Math.max(36, (tx - sx) / 2);
      return {
        id: link.parent_id + "->" + link.child_id,
        source: source.task,
        target: target.task,
        d: "M " + sx + " " + sy + " C " + middle + " " + sy + ", " + middle + " " + ty + ", " + tx + " " + ty,
      };
    }).filter(Boolean);

    return {
      nodes,
      edges,
      islands,
      width: graphWidth,
      height: Math.max(480, top),
      componentCount: components.length,
    };
  }


  function workflowIslandAction(island) {
    if (!island || island.isUnlinked) return null;
    return island.archiveId ? "restore" : "archive";
  }

  function workflowArchiveCanSubmit(preview, busy, acknowledged) {
    return !!preview && !busy && acknowledged === true;
  }

  function workflowArchiveCounts(preview) {
    const counts = preview && preview.counts ? preview.counts : {};
    return [
      ["total", "Total"],
      ["active", "Active"],
      ["running", "Running"],
      ["review", "Review"],
      ["done", "Done"],
      ["archived", "Archived"],
    ].filter(function (item) {
      return item[0] === "total" || Number(counts[item[0]] || 0) > 0;
    }).map(function (item) {
      return { key: item[0], label: item[1], value: Number(counts[item[0]] || 0) };
    });
  }

  return {
    GRAPH_STATUS_LABELS,
    beginLayoutSettingsRequest,
    buildTaskGraphLayout,
    graphCountLabel,
    invalidateLayoutSettingsRequests,
    isCurrentLayoutSettingsRequest,
    workflowArchiveCanSubmit,
    workflowArchiveCounts,
    workflowIslandAction,
  };
});
