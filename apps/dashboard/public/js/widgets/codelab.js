// Code Lab — an in-browser coding practice box. You write a JS function named
// `solution`; Run executes it against test cases inside a sandboxed Web Worker
// (with a timeout, so infinite loops can't hang the page). Progressive hints and
// a reveal-solution help you learn. All local — no network, works offline.

import { h, clear, toast } from "../utils.js";

const PROBLEMS = [
  {
    id: "twosum", title: "Two Sum", topic: "Arrays", difficulty: "Easy",
    prompt: "Return the indices of the two numbers in `nums` that add up to `target`. Exactly one solution; don't reuse an index.\n\nsolution(nums, target) → [i, j]",
    starter: "function solution(nums, target) {\n  // your code\n}\n",
    tests: [
      { name: "[2,7,11,15], 9", args: [[2, 7, 11, 15], 9], expect: [0, 1] },
      { name: "[3,2,4], 6", args: [[3, 2, 4], 6], expect: [1, 2] },
      { name: "[3,3], 6", args: [[3, 3], 6], expect: [0, 1] },
    ],
    hints: ["Brute force is two nested loops — O(n²). Can you do one pass?",
      "Store each value's index in a Map as you go.",
      "For each number x, check if (target − x) is already in the Map."],
    solution: "function solution(nums, target) {\n  const seen = new Map();\n  for (let i = 0; i < nums.length; i++) {\n    const need = target - nums[i];\n    if (seen.has(need)) return [seen.get(need), i];\n    seen.set(nums[i], i);\n  }\n}",
  },
  {
    id: "reverse", title: "Reverse a string", topic: "Strings", difficulty: "Easy",
    prompt: "Return `s` reversed.\n\nsolution(s) → string",
    starter: "function solution(s) {\n\n}\n",
    tests: [
      { name: '"hello"', args: ["hello"], expect: "olleh" },
      { name: '"a"', args: ["a"], expect: "a" },
      { name: '""', args: [""], expect: "" },
    ],
    hints: ["Strings have no reverse(), but arrays do.", "split → reverse → join."],
    solution: "function solution(s) {\n  return s.split('').reverse().join('');\n}",
  },
  {
    id: "fizzbuzz", title: "FizzBuzz", topic: "Logic", difficulty: "Easy",
    prompt: "Return an array 1..n where multiples of 3 are 'Fizz', of 5 'Buzz', of both 'FizzBuzz', else the number.\n\nsolution(n) → array",
    starter: "function solution(n) {\n\n}\n",
    tests: [
      { name: "5", args: [5], expect: [1, 2, "Fizz", 4, "Buzz"] },
      { name: "15 last", args: [15], expect: [1, 2, "Fizz", 4, "Buzz", "Fizz", 7, 8, "Fizz", "Buzz", 11, "Fizz", 13, 14, "FizzBuzz"] },
    ],
    hints: ["Check %15 (or %3 && %5) first.", "Build the array with a loop from 1 to n."],
    solution: "function solution(n) {\n  const out = [];\n  for (let i = 1; i <= n; i++) {\n    if (i % 15 === 0) out.push('FizzBuzz');\n    else if (i % 3 === 0) out.push('Fizz');\n    else if (i % 5 === 0) out.push('Buzz');\n    else out.push(i);\n  }\n  return out;\n}",
  },
  {
    id: "palindrome", title: "Valid palindrome", topic: "Strings", difficulty: "Easy",
    prompt: "Return true if `s` reads the same forwards and backwards, ignoring case and non-alphanumeric characters.\n\nsolution(s) → boolean",
    starter: "function solution(s) {\n\n}\n",
    tests: [
      { name: '"A man, a plan..."', args: ["A man, a plan, a canal: Panama"], expect: true },
      { name: '"race a car"', args: ["race a car"], expect: false },
      { name: '""', args: [""], expect: true },
    ],
    hints: ["Normalise first: lowercase and strip non-alphanumerics with a regex.",
      "Compare the cleaned string to its reverse.", "/[^a-z0-9]/g after toLowerCase()."],
    solution: "function solution(s) {\n  const c = s.toLowerCase().replace(/[^a-z0-9]/g, '');\n  return c === c.split('').reverse().join('');\n}",
  },
  {
    id: "fib", title: "Fibonacci", topic: "Recursion / DP", difficulty: "Easy",
    prompt: "Return the n-th Fibonacci number (0-indexed: 0,1,1,2,3,5,…). Aim for O(n), not exponential recursion.\n\nsolution(n) → number",
    starter: "function solution(n) {\n\n}\n",
    tests: [
      { name: "0", args: [0], expect: 0 }, { name: "10", args: [10], expect: 55 },
      { name: "30", args: [30], expect: 832040 },
    ],
    hints: ["Naive recursion recomputes the same values — exponential.",
      "Iterate, keeping the last two values.", "a=0, b=1; loop n times swapping."],
    solution: "function solution(n) {\n  let a = 0, b = 1;\n  for (let i = 0; i < n; i++) [a, b] = [b, a + b];\n  return a;\n}",
  },
  {
    id: "flatten", title: "Flatten nested array", topic: "Recursion", difficulty: "Medium",
    prompt: "Flatten an arbitrarily nested array of numbers into a single-level array (order preserved).\n\nsolution(arr) → array",
    starter: "function solution(arr) {\n\n}\n",
    tests: [
      { name: "[1,[2,[3,[4]]]]", args: [[1, [2, [3, [4]]]]], expect: [1, 2, 3, 4] },
      { name: "[[1,2],[3,[4,5]]]", args: [[[1, 2], [3, [4, 5]]]], expect: [1, 2, 3, 4, 5] },
    ],
    hints: ["Recurse: if an element is an array, flatten it too.",
      "Array.isArray(x) tells you when to recurse.", "reduce + concat, or a helper that pushes."],
    solution: "function solution(arr) {\n  return arr.reduce((acc, x) =>\n    acc.concat(Array.isArray(x) ? solution(x) : x), []);\n}",
  },
  {
    id: "kadane", title: "Maximum subarray sum", topic: "Dynamic programming", difficulty: "Medium",
    prompt: "Return the largest sum of any contiguous subarray of `nums` (at least one element).\n\nsolution(nums) → number",
    starter: "function solution(nums) {\n\n}\n",
    tests: [
      { name: "[-2,1,-3,4,-1,2,1,-5,4]", args: [[-2, 1, -3, 4, -1, 2, 1, -5, 4]], expect: 6 },
      { name: "[1]", args: [[1]], expect: 1 },
      { name: "[-3,-1,-2]", args: [[-3, -1, -2]], expect: -1 },
    ],
    hints: ["Kadane's algorithm — O(n).",
      "At each element, either extend the running sum or start fresh at this element.",
      "cur = max(x, cur + x); best = max(best, cur)."],
    solution: "function solution(nums) {\n  let cur = nums[0], best = nums[0];\n  for (let i = 1; i < nums.length; i++) {\n    cur = Math.max(nums[i], cur + nums[i]);\n    best = Math.max(best, cur);\n  }\n  return best;\n}",
  },
  {
    id: "bsearch", title: "Binary search", topic: "Algorithms", difficulty: "Medium",
    prompt: "Return the index of `target` in the sorted array `nums`, or -1 if absent. O(log n).\n\nsolution(nums, target) → number",
    starter: "function solution(nums, target) {\n\n}\n",
    tests: [
      { name: "[-1,0,3,5,9,12], 9", args: [[-1, 0, 3, 5, 9, 12], 9], expect: 4 },
      { name: "[-1,0,3,5,9,12], 2", args: [[-1, 0, 3, 5, 9, 12], 2], expect: -1 },
    ],
    hints: ["Track lo and hi pointers.", "mid = (lo+hi)>>1; compare and halve the range.",
      "Careful with lo <= hi and updating mid ± 1."],
    solution: "function solution(nums, target) {\n  let lo = 0, hi = nums.length - 1;\n  while (lo <= hi) {\n    const mid = (lo + hi) >> 1;\n    if (nums[mid] === target) return mid;\n    if (nums[mid] < target) lo = mid + 1; else hi = mid - 1;\n  }\n  return -1;\n}",
  },
  {
    id: "anagram", title: "Group anagrams", topic: "Hash maps", difficulty: "Medium",
    prompt: "Group the words that are anagrams of each other. Return an array of groups; within each group keep input order; groups ordered by first appearance.\n\nsolution(words) → array of arrays",
    starter: "function solution(words) {\n\n}\n",
    tests: [
      { name: '["eat","tea","tan","ate","nat","bat"]', args: [["eat", "tea", "tan", "ate", "nat", "bat"]], expect: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]] },
    ],
    hints: ["Anagrams share the same sorted letters — use that as a key.",
      "Map key → array of words.", "key = word.split('').sort().join('')."],
    solution: "function solution(words) {\n  const m = new Map();\n  for (const w of words) {\n    const k = w.split('').sort().join('');\n    if (!m.has(k)) m.set(k, []);\n    m.get(k).push(w);\n  }\n  return [...m.values()];\n}",
  },
];

const WORKER_SRC = `self.onmessage = (e) => {
  const { code, tests } = e.data;
  let solution;
  try {
    solution = (new Function(code + "\\n;return typeof solution!=='undefined'?solution:undefined;"))();
  } catch (err) { self.postMessage({ compileError: String(err) }); return; }
  if (typeof solution !== 'function') { self.postMessage({ compileError: 'Define a function named solution.' }); return; }
  const results = tests.map((t) => {
    try {
      const got = solution.apply(null, t.args);
      return { name: t.name, pass: JSON.stringify(got) === JSON.stringify(t.expect), got: JSON.stringify(got) };
    } catch (err) { return { name: t.name, pass: false, got: 'Error: ' + err.message }; }
  });
  self.postMessage({ results });
};`;

export default {
  type: "codelab",
  title: "Code Lab",
  icon: "⌨️",
  defaultSize: "l",

  render(body, ctx) {
    const { store } = ctx;
    const st = () => store.state.codelab || {};
    let active = st().active || PROBLEMS[0].id;
    let hintsShown = 0;
    let showSolution = false;

    const codeFor = (id) => (st().code || {})[id];
    const saveCode = (id, code) => store.update((s) => {
      if (!s.codelab) s.codelab = {};
      if (!s.codelab.code) s.codelab.code = {};
      s.codelab.code[id] = code;
      s.codelab.active = id;
    }, "codelab");

    const runTests = (problem, code, out) => {
      clear(out).append(h("div.muted.small", {}, "Running…"));
      let worker; let done = false;
      try {
        worker = new Worker(URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" })));
      } catch {
        clear(out).append(h("div.widget-error", {}, "Web Workers unavailable in this browser."));
        return;
      }
      const timer = setTimeout(() => {
        if (done) return; done = true; worker.terminate();
        clear(out).append(h("div.widget-error", {}, "⏱ Timed out (possible infinite loop)."));
      }, 2500);
      worker.onmessage = (e) => {
        if (done) return; done = true; clearTimeout(timer); worker.terminate();
        const { compileError, results } = e.data;
        if (compileError) { clear(out).append(h("div.widget-error", {}, compileError)); return; }
        const passed = results.filter((r) => r.pass).length;
        clear(out).append(
          h("div.cl-summary", { class: passed === results.length ? "cl-summary cl-pass" : "cl-summary cl-fail" },
            `${passed}/${results.length} tests passed${passed === results.length ? " ✓" : ""}`),
          ...results.map((r) => h("div.cl-test", { class: r.pass ? "cl-test cl-ok" : "cl-test cl-no" },
            h("span.cl-test-icon", {}, r.pass ? "✓" : "✗"),
            h("span.cl-test-name", {}, r.name),
            r.pass ? null : h("span.muted.small.cl-test-got", {}, `got ${r.got}`))));
        if (passed === results.length) toast("All tests passed 🎉");
      };
      worker.postMessage({ code, tests: problem.tests });
    };

    const draw = () => {
      const problem = PROBLEMS.find((p) => p.id === active) || PROBLEMS[0];
      hintsShown = 0; showSolution = false;

      const picker = h("select.select.cl-picker", {
        "aria-label": "Problem",
        onchange: (ev) => { active = ev.target.value; store.update((s) => { (s.codelab || (s.codelab = {})).active = active; }, "codelab"); draw(); },
      }, PROBLEMS.map((p) => h("option", { value: p.id, selected: p.id === active }, `${p.title} · ${p.difficulty}`)));

      const editor = h("textarea.cl-editor", {
        spellcheck: "false", rows: 10, "aria-label": "Code editor",
        oninput: (ev) => saveCode(problem.id, ev.target.value),
        onkeydown: (ev) => {
          if (ev.key === "Tab") { ev.preventDefault(); const t = ev.target, s = t.selectionStart;
            t.value = t.value.slice(0, s) + "  " + t.value.slice(t.selectionEnd);
            t.selectionStart = t.selectionEnd = s + 2; saveCode(problem.id, t.value); }
        },
      }, codeFor(problem.id) ?? problem.starter);

      const output = h("div.cl-output");
      const hintBox = h("div.cl-hints");
      const renderHints = () => {
        clear(hintBox);
        for (let i = 0; i < hintsShown; i++) hintBox.append(h("div.cl-hint", {}, `💡 ${problem.hints[i]}`));
        if (showSolution) hintBox.append(h("pre.cl-solution", {}, problem.solution));
      };

      clear(body).append(
        h("div.cl-head", {}, picker,
          h("span.cl-topic.muted.small", {}, `${problem.topic}`)),
        h("div.cl-prompt", {}, problem.prompt),
        editor,
        h("div.cl-actions", {},
          h("button.btn.btn-primary", { type: "button", onclick: () => runTests(problem, editor.value, output) }, "▶ Run tests"),
          h("button.btn", { type: "button", onclick: () => {
            if (hintsShown < problem.hints.length) { hintsShown++; renderHints(); }
            else toast("No more hints — try Solution");
          } }, "💡 Hint"),
          h("button.btn", { type: "button", onclick: () => { showSolution = !showSolution; renderHints(); } }, "Solution"),
          h("button.link-btn", { type: "button", onclick: () => { editor.value = problem.starter; saveCode(problem.id, problem.starter); clear(output); } }, "reset")),
        output, hintBox,
        h("div.muted.small.cl-note", {}, "Runs your JS in a sandboxed worker · define a function named `solution`."),
      );
    };

    ctx.onRefresh(draw);
    draw();
  },
};
