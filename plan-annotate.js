/**
 * plan-annotate.js — In-page annotation module
 * Self-contained, no dependencies.
 * Usage: include via &lt;script src="plan-annotate.js" defer&gt;&lt;/script&gt;
 */

(function () {
  "use strict";

  let comments = [];
  let activeBox = null;

  function getSelectionInfo() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return null;
    const range = sel.getRangeAt(0);
    const text = sel.toString().trim();
    if (!text) return null;
    const section = findSection(range);
    return {
      section,
      text,
      start_offset: range.startOffset,
      end_offset: range.endOffset,
    };
  }

  function findSection(range) {
    let el = range.commonAncestorContainer;
    while (el && el !== document.body) {
      if (el.tagName === "H1" || el.tagName === "H2" || el.tagName === "H3") {
        return el.textContent.trim();
      }
      el = el.parentElement;
    }
    return "Unnamed";
  }

  function posFromRange(range) {
    var rect = null;
    if (typeof range.getBoundingClientRect === "function") {
      rect = range.getBoundingClientRect();
    }
    if (!rect && range.startContainer && range.startContainer.parentElement) {
      rect = range.startContainer.parentElement.getBoundingClientRect();
    }
    if (!rect) {
      return { x: 0, y: 0 };
    }
    return {
      x: rect.left + window.scrollX,
      y: rect.bottom + window.scrollY + 6,
    };
  }

  function showCommentBox(selInfo, range) {
    hideCommentBox();
    const pos = posFromRange(range);
    const box = document.createElement("div");
    box.className = "h-annot-box";
    box.style.left = pos.x + "px";
    box.style.top = pos.y + "px";
    box.innerHTML =
      '<textarea class="h-annot-textarea" rows="3" placeholder="Add a comment…"></textarea>' +
      '<div class="h-annot-box-actions">' +
      '<button class="h-annot-pin-btn">Pin</button>' +
      '<button class="h-annot-cancel-btn">Cancel</button>' +
      "</div>";
    document.body.appendChild(box);
    activeBox = { box, selInfo, range };
    const ta = box.querySelector("textarea");
    ta.focus();
    ta.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        pinComment();
      } else if (e.key === "Escape") {
        hideCommentBox();
      }
    });
    box.querySelector(".h-annot-pin-btn").addEventListener("click", pinComment);
    box.querySelector(".h-annot-cancel-btn").addEventListener("click", hideCommentBox);
  }

  function hideCommentBox() {
    if (activeBox) {
      activeBox.box.remove();
      activeBox = null;
    }
  }

  function pinComment() {
    if (!activeBox) return;
    const { box, selInfo, range } = activeBox;
    const textarea = box.querySelector("textarea");
    const commentText = textarea.value.trim();
    if (!commentText) return;

    const comment = {
      section: selInfo.section,
      text: selInfo.text,
      start_offset: selInfo.start_offset,
      end_offset: selInfo.end_offset,
      comment: commentText,
      author: "user",
      status: "open",
      timestamp: new Date().toISOString(),
    };
    const id = comments.length;
    comment.id = id;
    comments.push(comment);

    // Wrap selection in highlight span — use stored range from activeBox
    var r = activeBox ? activeBox.range : null;
    if (!r && window.getSelection().rangeCount) {
      r = window.getSelection().getRangeAt(0);
    }
    if (r) {
      var hl = document.createElement("span");
      hl.className = "h-annot-hl";
      hl.dataset.annotId = id;
      try {
        r.surroundContents(hl);
      } catch (_) {
        // fallback: extract and re-wrap
        var frag = r.extractContents();
        hl.appendChild(frag);
        r.insertNode(hl);
      }
    }

    hideCommentBox();
    clearSelection();
    renderCallout(id, comment);
  }

  function renderCallout(id, comment) {
    const hl = document.querySelector('.h-annot-hl[data-annot-id="' + id + '"]');
    if (!hl) return;
    const callout = document.createElement("div");
    callout.className = "h-annot-callout";
    callout.dataset.annotId = id;
    callout.innerHTML =
      '<span class="h-annot-callout-author">' +
      htmlEscape(comment.author) +
      "</span>: " +
      htmlEscape(comment.comment) +
      ' <span class="h-annot-callout-status">[' +
      htmlEscape(comment.status) +
      "]</span>" +
      '<span class="h-annot-callout-actions">' +
      '<button class="h-annot-edit-btn" title="Edit">&#9998;</button>' +
      '<button class="h-annot-delete-btn" title="Delete">&times;</button>' +
      "</span>";
    hl.parentNode.insertBefore(callout, hl.nextSibling);

    callout.querySelector(".h-annot-edit-btn").addEventListener("click", function (e) {
      e.stopPropagation();
      editCallout(id);
    });
    callout.querySelector(".h-annot-delete-btn").addEventListener("click", function (e) {
      e.stopPropagation();
      deleteCallout(id);
    });
  }

  function editCallout(id) {
    const comment = comments[id];
    if (!comment) return;
    const newText = prompt("Edit comment:", comment.comment);
    if (newText !== null && newText.trim()) {
      comment.comment = newText.trim();
      comment.timestamp = new Date().toISOString();
      const callout = document.querySelector('.h-annot-callout[data-annot-id="' + id + '"]');
      if (callout) {
        const textNode = callout.childNodes[2];
        if (textNode) textNode.textContent = ": " + comment.comment + " ";
      }
    }
  }

  function deleteCallout(id) {
    if (!confirm("Delete this annotation?")) return;
    const hl = document.querySelector('.h-annot-hl[data-annot-id="' + id + '"]');
    const callout = document.querySelector('.h-annot-callout[data-annot-id="' + id + '"]');
    if (hl) {
      const parent = hl.parentNode;
      while (hl.firstChild) parent.insertBefore(hl.firstChild, hl);
      parent.removeChild(hl);
    }
    if (callout) callout.remove();
    comments[id] = null;
  }

  function clearSelection() {
    const sel = window.getSelection();
    if (sel) sel.removeAllRanges();
  }

  function htmlEscape(str) {
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  function exportComments() {
    if (comments.every(function (c) { return c === null; })) return;
    const data = {
      plan: "plan-annotate",
      exported_at: new Date().toISOString(),
      comments: comments.filter(Boolean).map(function (c) {
        return {
          id: c.id,
          section: c.section,
          text: c.text,
          start_offset: c.start_offset,
          end_offset: c.end_offset,
          comment: c.comment,
          author: c.author,
          status: c.status,
          timestamp: c.timestamp,
        };
      }),
    };
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "plan-annotate.json";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    try { URL.revokeObjectURL(url); } catch (_) {}
  }

  function init() {
    const header = document.querySelector("header, h1, h2, .header, nav");
    if (header) {
      const exportBtn = document.createElement("button");
      exportBtn.className = "h-annot-export-btn";
      exportBtn.textContent = "Export comments";
      exportBtn.addEventListener("click", exportComments);
      header.appendChild(exportBtn);
    }

    document.body.addEventListener("mouseup", function (e) {
      // Let clicks inside the comment box finish first
      if (activeBox && activeBox.box.contains(e.target)) return;

      const selInfo = getSelectionInfo();
      if (!selInfo) {
        hideCommentBox();
        return;
      }
      const sel = window.getSelection();
      const range = sel.getRangeAt(0);
      showCommentBox(selInfo, range);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
