import { createRequire } from 'module'; const require = createRequire(import.meta.url);
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __require = /* @__PURE__ */ ((x2) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x2, {
  get: (a, b2) => (typeof require !== "undefined" ? require : a)[b2]
}) : x2)(function(x2) {
  if (typeof require !== "undefined") return require.apply(this, arguments);
  throw Error('Dynamic require of "' + x2 + '" is not supported');
});
var __esm = (fn, res, err) => function __init() {
  if (err) throw err[0];
  try {
    return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
  } catch (e) {
    throw err = [e], e;
  }
};
var __commonJS = (cb, mod) => function __require2() {
  try {
    return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
  } catch (e) {
    throw mod = 0, e;
  }
};
var __export = (target2, all) => {
  for (var name in all)
    __defProp(target2, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target2) => (target2 = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target2, "default", { value: mod, enumerable: true }) : target2,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// ../../node_modules/ms/index.js
var require_ms = __commonJS({
  "../../node_modules/ms/index.js"(exports, module) {
    var s = 1e3;
    var m = s * 60;
    var h2 = m * 60;
    var d = h2 * 24;
    var w = d * 7;
    var y2 = d * 365.25;
    module.exports = function(val, options) {
      options = options || {};
      var type = typeof val;
      if (type === "string" && val.length > 0) {
        return parse(val);
      } else if (type === "number" && isFinite(val)) {
        return options.long ? fmtLong(val) : fmtShort(val);
      }
      throw new Error(
        "val is not a non-empty string or a valid number. val=" + JSON.stringify(val)
      );
    };
    function parse(str) {
      str = String(str);
      if (str.length > 100) {
        return;
      }
      var match = /^(-?(?:\d+)?\.?\d+) *(milliseconds?|msecs?|ms|seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|w|years?|yrs?|y)?$/i.exec(
        str
      );
      if (!match) {
        return;
      }
      var n = parseFloat(match[1]);
      var type = (match[2] || "ms").toLowerCase();
      switch (type) {
        case "years":
        case "year":
        case "yrs":
        case "yr":
        case "y":
          return n * y2;
        case "weeks":
        case "week":
        case "w":
          return n * w;
        case "days":
        case "day":
        case "d":
          return n * d;
        case "hours":
        case "hour":
        case "hrs":
        case "hr":
        case "h":
          return n * h2;
        case "minutes":
        case "minute":
        case "mins":
        case "min":
        case "m":
          return n * m;
        case "seconds":
        case "second":
        case "secs":
        case "sec":
        case "s":
          return n * s;
        case "milliseconds":
        case "millisecond":
        case "msecs":
        case "msec":
        case "ms":
          return n;
        default:
          return void 0;
      }
    }
    function fmtShort(ms) {
      var msAbs = Math.abs(ms);
      if (msAbs >= d) {
        return Math.round(ms / d) + "d";
      }
      if (msAbs >= h2) {
        return Math.round(ms / h2) + "h";
      }
      if (msAbs >= m) {
        return Math.round(ms / m) + "m";
      }
      if (msAbs >= s) {
        return Math.round(ms / s) + "s";
      }
      return ms + "ms";
    }
    function fmtLong(ms) {
      var msAbs = Math.abs(ms);
      if (msAbs >= d) {
        return plural(ms, msAbs, d, "day");
      }
      if (msAbs >= h2) {
        return plural(ms, msAbs, h2, "hour");
      }
      if (msAbs >= m) {
        return plural(ms, msAbs, m, "minute");
      }
      if (msAbs >= s) {
        return plural(ms, msAbs, s, "second");
      }
      return ms + " ms";
    }
    function plural(ms, msAbs, n, name) {
      var isPlural = msAbs >= n * 1.5;
      return Math.round(ms / n) + " " + name + (isPlural ? "s" : "");
    }
  }
});

// ../../node_modules/debug/src/common.js
var require_common = __commonJS({
  "../../node_modules/debug/src/common.js"(exports, module) {
    function setup(env2) {
      createDebug.debug = createDebug;
      createDebug.default = createDebug;
      createDebug.coerce = coerce;
      createDebug.disable = disable;
      createDebug.enable = enable;
      createDebug.enabled = enabled2;
      createDebug.humanize = require_ms();
      createDebug.destroy = destroy;
      Object.keys(env2).forEach((key) => {
        createDebug[key] = env2[key];
      });
      createDebug.names = [];
      createDebug.skips = [];
      createDebug.formatters = {};
      function selectColor(namespace) {
        let hash = 0;
        for (let i2 = 0; i2 < namespace.length; i2++) {
          hash = (hash << 5) - hash + namespace.charCodeAt(i2);
          hash |= 0;
        }
        return createDebug.colors[Math.abs(hash) % createDebug.colors.length];
      }
      createDebug.selectColor = selectColor;
      function createDebug(namespace) {
        let prevTime;
        let enableOverride = null;
        let namespacesCache;
        let enabledCache;
        function debug2(...args) {
          if (!debug2.enabled) {
            return;
          }
          const self = debug2;
          const curr = Number(/* @__PURE__ */ new Date());
          const ms = curr - (prevTime || curr);
          self.diff = ms;
          self.prev = prevTime;
          self.curr = curr;
          prevTime = curr;
          args[0] = createDebug.coerce(args[0]);
          if (typeof args[0] !== "string") {
            args.unshift("%O");
          }
          let index = 0;
          args[0] = args[0].replace(/%([a-zA-Z%])/g, (match, format) => {
            if (match === "%%") {
              return "%";
            }
            index++;
            const formatter = createDebug.formatters[format];
            if (typeof formatter === "function") {
              const val = args[index];
              match = formatter.call(self, val);
              args.splice(index, 1);
              index--;
            }
            return match;
          });
          createDebug.formatArgs.call(self, args);
          const logFn = self.log || createDebug.log;
          logFn.apply(self, args);
        }
        debug2.namespace = namespace;
        debug2.useColors = createDebug.useColors();
        debug2.color = createDebug.selectColor(namespace);
        debug2.extend = extend;
        debug2.destroy = createDebug.destroy;
        Object.defineProperty(debug2, "enabled", {
          enumerable: true,
          configurable: false,
          get: () => {
            if (enableOverride !== null) {
              return enableOverride;
            }
            if (namespacesCache !== createDebug.namespaces) {
              namespacesCache = createDebug.namespaces;
              enabledCache = createDebug.enabled(namespace);
            }
            return enabledCache;
          },
          set: (v) => {
            enableOverride = v;
          }
        });
        if (typeof createDebug.init === "function") {
          createDebug.init(debug2);
        }
        return debug2;
      }
      function extend(namespace, delimiter) {
        const newDebug = createDebug(this.namespace + (typeof delimiter === "undefined" ? ":" : delimiter) + namespace);
        newDebug.log = this.log;
        return newDebug;
      }
      function enable(namespaces) {
        createDebug.save(namespaces);
        createDebug.namespaces = namespaces;
        createDebug.names = [];
        createDebug.skips = [];
        const split = (typeof namespaces === "string" ? namespaces : "").trim().replace(/\s+/g, ",").split(",").filter(Boolean);
        for (const ns of split) {
          if (ns[0] === "-") {
            createDebug.skips.push(ns.slice(1));
          } else {
            createDebug.names.push(ns);
          }
        }
      }
      function matchesTemplate(search, template) {
        let searchIndex = 0;
        let templateIndex = 0;
        let starIndex = -1;
        let matchIndex = 0;
        while (searchIndex < search.length) {
          if (templateIndex < template.length && (template[templateIndex] === search[searchIndex] || template[templateIndex] === "*")) {
            if (template[templateIndex] === "*") {
              starIndex = templateIndex;
              matchIndex = searchIndex;
              templateIndex++;
            } else {
              searchIndex++;
              templateIndex++;
            }
          } else if (starIndex !== -1) {
            templateIndex = starIndex + 1;
            matchIndex++;
            searchIndex = matchIndex;
          } else {
            return false;
          }
        }
        while (templateIndex < template.length && template[templateIndex] === "*") {
          templateIndex++;
        }
        return templateIndex === template.length;
      }
      function disable() {
        const namespaces = [
          ...createDebug.names,
          ...createDebug.skips.map((namespace) => "-" + namespace)
        ].join(",");
        createDebug.enable("");
        return namespaces;
      }
      function enabled2(name) {
        for (const skip of createDebug.skips) {
          if (matchesTemplate(name, skip)) {
            return false;
          }
        }
        for (const ns of createDebug.names) {
          if (matchesTemplate(name, ns)) {
            return true;
          }
        }
        return false;
      }
      function coerce(val) {
        if (val instanceof Error) {
          return val.stack || val.message;
        }
        return val;
      }
      function destroy() {
        console.warn("Instance method `debug.destroy()` is deprecated and no longer does anything. It will be removed in the next major version of `debug`.");
      }
      createDebug.enable(createDebug.load());
      return createDebug;
    }
    module.exports = setup;
  }
});

// ../../node_modules/debug/src/browser.js
var require_browser = __commonJS({
  "../../node_modules/debug/src/browser.js"(exports, module) {
    exports.formatArgs = formatArgs;
    exports.save = save;
    exports.load = load;
    exports.useColors = useColors;
    exports.storage = localstorage();
    exports.destroy = /* @__PURE__ */ (() => {
      let warned = false;
      return () => {
        if (!warned) {
          warned = true;
          console.warn("Instance method `debug.destroy()` is deprecated and no longer does anything. It will be removed in the next major version of `debug`.");
        }
      };
    })();
    exports.colors = [
      "#0000CC",
      "#0000FF",
      "#0033CC",
      "#0033FF",
      "#0066CC",
      "#0066FF",
      "#0099CC",
      "#0099FF",
      "#00CC00",
      "#00CC33",
      "#00CC66",
      "#00CC99",
      "#00CCCC",
      "#00CCFF",
      "#3300CC",
      "#3300FF",
      "#3333CC",
      "#3333FF",
      "#3366CC",
      "#3366FF",
      "#3399CC",
      "#3399FF",
      "#33CC00",
      "#33CC33",
      "#33CC66",
      "#33CC99",
      "#33CCCC",
      "#33CCFF",
      "#6600CC",
      "#6600FF",
      "#6633CC",
      "#6633FF",
      "#66CC00",
      "#66CC33",
      "#9900CC",
      "#9900FF",
      "#9933CC",
      "#9933FF",
      "#99CC00",
      "#99CC33",
      "#CC0000",
      "#CC0033",
      "#CC0066",
      "#CC0099",
      "#CC00CC",
      "#CC00FF",
      "#CC3300",
      "#CC3333",
      "#CC3366",
      "#CC3399",
      "#CC33CC",
      "#CC33FF",
      "#CC6600",
      "#CC6633",
      "#CC9900",
      "#CC9933",
      "#CCCC00",
      "#CCCC33",
      "#FF0000",
      "#FF0033",
      "#FF0066",
      "#FF0099",
      "#FF00CC",
      "#FF00FF",
      "#FF3300",
      "#FF3333",
      "#FF3366",
      "#FF3399",
      "#FF33CC",
      "#FF33FF",
      "#FF6600",
      "#FF6633",
      "#FF9900",
      "#FF9933",
      "#FFCC00",
      "#FFCC33"
    ];
    function useColors() {
      if (typeof window !== "undefined" && window.process && (window.process.type === "renderer" || window.process.__nwjs)) {
        return true;
      }
      if (typeof navigator !== "undefined" && navigator.userAgent && navigator.userAgent.toLowerCase().match(/(edge|trident)\/(\d+)/)) {
        return false;
      }
      let m;
      return typeof document !== "undefined" && document.documentElement && document.documentElement.style && document.documentElement.style.WebkitAppearance || // Is firebug? http://stackoverflow.com/a/398120/376773
      typeof window !== "undefined" && window.console && (window.console.firebug || window.console.exception && window.console.table) || // Is firefox >= v31?
      // https://developer.mozilla.org/en-US/docs/Tools/Web_Console#Styling_messages
      typeof navigator !== "undefined" && navigator.userAgent && (m = navigator.userAgent.toLowerCase().match(/firefox\/(\d+)/)) && parseInt(m[1], 10) >= 31 || // Double check webkit in userAgent just in case we are in a worker
      typeof navigator !== "undefined" && navigator.userAgent && navigator.userAgent.toLowerCase().match(/applewebkit\/(\d+)/);
    }
    function formatArgs(args) {
      args[0] = (this.useColors ? "%c" : "") + this.namespace + (this.useColors ? " %c" : " ") + args[0] + (this.useColors ? "%c " : " ") + "+" + module.exports.humanize(this.diff);
      if (!this.useColors) {
        return;
      }
      const c3 = "color: " + this.color;
      args.splice(1, 0, c3, "color: inherit");
      let index = 0;
      let lastC = 0;
      args[0].replace(/%[a-zA-Z%]/g, (match) => {
        if (match === "%%") {
          return;
        }
        index++;
        if (match === "%c") {
          lastC = index;
        }
      });
      args.splice(lastC, 0, c3);
    }
    exports.log = console.debug || console.log || (() => {
    });
    function save(namespaces) {
      try {
        if (namespaces) {
          exports.storage.setItem("debug", namespaces);
        } else {
          exports.storage.removeItem("debug");
        }
      } catch (error) {
      }
    }
    function load() {
      let r2;
      try {
        r2 = exports.storage.getItem("debug") || exports.storage.getItem("DEBUG");
      } catch (error) {
      }
      if (!r2 && typeof process !== "undefined" && "env" in process) {
        r2 = process.env.DEBUG;
      }
      return r2;
    }
    function localstorage() {
      try {
        return localStorage;
      } catch (error) {
      }
    }
    module.exports = require_common()(exports);
    var { formatters } = module.exports;
    formatters.j = function(v) {
      try {
        return JSON.stringify(v);
      } catch (error) {
        return "[UnexpectedJSONParseError]: " + error.message;
      }
    };
  }
});

// ../../node_modules/supports-color/index.js
var supports_color_exports = {};
__export(supports_color_exports, {
  createSupportsColor: () => createSupportsColor,
  default: () => supports_color_default
});
import process2 from "node:process";
import os3 from "node:os";
import tty from "node:tty";
function hasFlag(flag, argv = globalThis.Deno ? globalThis.Deno.args : process2.argv) {
  const prefix = flag.startsWith("-") ? "" : flag.length === 1 ? "-" : "--";
  const position = argv.indexOf(prefix + flag);
  const terminatorPosition = argv.indexOf("--");
  return position !== -1 && (terminatorPosition === -1 || position < terminatorPosition);
}
function envForceColor() {
  if (!("FORCE_COLOR" in env)) {
    return;
  }
  if (env.FORCE_COLOR === "true") {
    return 1;
  }
  if (env.FORCE_COLOR === "false") {
    return 0;
  }
  if (env.FORCE_COLOR.length === 0) {
    return 1;
  }
  const level = Math.min(Number.parseInt(env.FORCE_COLOR, 10), 3);
  if (![0, 1, 2, 3].includes(level)) {
    return;
  }
  return level;
}
function translateLevel(level) {
  if (level === 0) {
    return false;
  }
  return {
    level,
    hasBasic: true,
    has256: level >= 2,
    has16m: level >= 3
  };
}
function _supportsColor(haveStream, { streamIsTTY, sniffFlags = true } = {}) {
  const noFlagForceColor = envForceColor();
  if (noFlagForceColor !== void 0) {
    flagForceColor = noFlagForceColor;
  }
  const forceColor = sniffFlags ? flagForceColor : noFlagForceColor;
  if (forceColor === 0) {
    return 0;
  }
  if (sniffFlags) {
    if (hasFlag("color=16m") || hasFlag("color=full") || hasFlag("color=truecolor")) {
      return 3;
    }
    if (hasFlag("color=256")) {
      return 2;
    }
  }
  if ("TF_BUILD" in env && "AGENT_NAME" in env) {
    return 1;
  }
  if (haveStream && !streamIsTTY && forceColor === void 0) {
    return 0;
  }
  const min = forceColor || 0;
  if (env.TERM === "dumb") {
    return min;
  }
  if (process2.platform === "win32") {
    const osRelease = os3.release().split(".");
    if (Number(osRelease[0]) >= 10 && Number(osRelease[2]) >= 10586) {
      return Number(osRelease[2]) >= 14931 ? 3 : 2;
    }
    return 1;
  }
  if ("CI" in env) {
    if (["GITHUB_ACTIONS", "GITEA_ACTIONS", "CIRCLECI"].some((key) => key in env)) {
      return 3;
    }
    if (["TRAVIS", "APPVEYOR", "GITLAB_CI", "BUILDKITE", "DRONE"].some((sign) => sign in env) || env.CI_NAME === "codeship") {
      return 1;
    }
    return min;
  }
  if ("TEAMCITY_VERSION" in env) {
    return /^(9\.(0*[1-9]\d*)\.|\d{2,}\.)/.test(env.TEAMCITY_VERSION) ? 1 : 0;
  }
  if (env.COLORTERM === "truecolor") {
    return 3;
  }
  if (env.TERM === "xterm-kitty") {
    return 3;
  }
  if (env.TERM === "xterm-ghostty") {
    return 3;
  }
  if (env.TERM === "wezterm") {
    return 3;
  }
  if ("TERM_PROGRAM" in env) {
    const version = Number.parseInt((env.TERM_PROGRAM_VERSION || "").split(".")[0], 10);
    switch (env.TERM_PROGRAM) {
      case "iTerm.app": {
        return version >= 3 ? 3 : 2;
      }
      case "Apple_Terminal": {
        return 2;
      }
    }
  }
  if (/-256(color)?$/i.test(env.TERM)) {
    return 2;
  }
  if (/^screen|^xterm|^vt100|^vt220|^rxvt|color|ansi|cygwin|linux/i.test(env.TERM)) {
    return 1;
  }
  if ("COLORTERM" in env) {
    return 1;
  }
  return min;
}
function createSupportsColor(stream, options = {}) {
  const level = _supportsColor(stream, {
    streamIsTTY: stream && stream.isTTY,
    ...options
  });
  return translateLevel(level);
}
var env, flagForceColor, supportsColor, supports_color_default;
var init_supports_color = __esm({
  "../../node_modules/supports-color/index.js"() {
    ({ env } = process2);
    if (hasFlag("no-color") || hasFlag("no-colors") || hasFlag("color=false") || hasFlag("color=never")) {
      flagForceColor = 0;
    } else if (hasFlag("color") || hasFlag("colors") || hasFlag("color=true") || hasFlag("color=always")) {
      flagForceColor = 1;
    }
    supportsColor = {
      stdout: createSupportsColor({ isTTY: tty.isatty(1) }),
      stderr: createSupportsColor({ isTTY: tty.isatty(2) })
    };
    supports_color_default = supportsColor;
  }
});

// ../../node_modules/debug/src/node.js
var require_node = __commonJS({
  "../../node_modules/debug/src/node.js"(exports, module) {
    var tty2 = __require("tty");
    var util = __require("util");
    exports.init = init;
    exports.log = log;
    exports.formatArgs = formatArgs;
    exports.save = save;
    exports.load = load;
    exports.useColors = useColors;
    exports.destroy = util.deprecate(
      () => {
      },
      "Instance method `debug.destroy()` is deprecated and no longer does anything. It will be removed in the next major version of `debug`."
    );
    exports.colors = [6, 2, 3, 4, 5, 1];
    try {
      const supportsColor2 = (init_supports_color(), __toCommonJS(supports_color_exports));
      if (supportsColor2 && (supportsColor2.stderr || supportsColor2).level >= 2) {
        exports.colors = [
          20,
          21,
          26,
          27,
          32,
          33,
          38,
          39,
          40,
          41,
          42,
          43,
          44,
          45,
          56,
          57,
          62,
          63,
          68,
          69,
          74,
          75,
          76,
          77,
          78,
          79,
          80,
          81,
          92,
          93,
          98,
          99,
          112,
          113,
          128,
          129,
          134,
          135,
          148,
          149,
          160,
          161,
          162,
          163,
          164,
          165,
          166,
          167,
          168,
          169,
          170,
          171,
          172,
          173,
          178,
          179,
          184,
          185,
          196,
          197,
          198,
          199,
          200,
          201,
          202,
          203,
          204,
          205,
          206,
          207,
          208,
          209,
          214,
          215,
          220,
          221
        ];
      }
    } catch (error) {
    }
    exports.inspectOpts = Object.keys(process.env).filter((key) => {
      return /^debug_/i.test(key);
    }).reduce((obj, key) => {
      const prop = key.substring(6).toLowerCase().replace(/_([a-z])/g, (_2, k2) => {
        return k2.toUpperCase();
      });
      let val = process.env[key];
      if (/^(yes|on|true|enabled)$/i.test(val)) {
        val = true;
      } else if (/^(no|off|false|disabled)$/i.test(val)) {
        val = false;
      } else if (val === "null") {
        val = null;
      } else {
        val = Number(val);
      }
      obj[prop] = val;
      return obj;
    }, {});
    function useColors() {
      return "colors" in exports.inspectOpts ? Boolean(exports.inspectOpts.colors) : tty2.isatty(process.stderr.fd);
    }
    function formatArgs(args) {
      const { namespace: name, useColors: useColors2 } = this;
      if (useColors2) {
        const c3 = this.color;
        const colorCode = "\x1B[3" + (c3 < 8 ? c3 : "8;5;" + c3);
        const prefix = `  ${colorCode};1m${name} \x1B[0m`;
        args[0] = prefix + args[0].split("\n").join("\n" + prefix);
        args.push(colorCode + "m+" + module.exports.humanize(this.diff) + "\x1B[0m");
      } else {
        args[0] = getDate() + name + " " + args[0];
      }
    }
    function getDate() {
      if (exports.inspectOpts.hideDate) {
        return "";
      }
      return (/* @__PURE__ */ new Date()).toISOString() + " ";
    }
    function log(...args) {
      return process.stderr.write(util.formatWithOptions(exports.inspectOpts, ...args) + "\n");
    }
    function save(namespaces) {
      if (namespaces) {
        process.env.DEBUG = namespaces;
      } else {
        delete process.env.DEBUG;
      }
    }
    function load() {
      return process.env.DEBUG;
    }
    function init(debug2) {
      debug2.inspectOpts = {};
      const keys = Object.keys(exports.inspectOpts);
      for (let i2 = 0; i2 < keys.length; i2++) {
        debug2.inspectOpts[keys[i2]] = exports.inspectOpts[keys[i2]];
      }
    }
    module.exports = require_common()(exports);
    var { formatters } = module.exports;
    formatters.o = function(v) {
      this.inspectOpts.colors = this.useColors;
      return util.inspect(v, this.inspectOpts).split("\n").map((str) => str.trim()).join(" ");
    };
    formatters.O = function(v) {
      this.inspectOpts.colors = this.useColors;
      return util.inspect(v, this.inspectOpts);
    };
  }
});

// ../../node_modules/debug/src/index.js
var require_src = __commonJS({
  "../../node_modules/debug/src/index.js"(exports, module) {
    if (typeof process === "undefined" || process.type === "renderer" || process.browser === true || process.__nwjs) {
      module.exports = require_browser();
    } else {
      module.exports = require_node();
    }
  }
});

// ../../node_modules/@kwsites/file-exists/dist/src/index.js
var require_src2 = __commonJS({
  "../../node_modules/@kwsites/file-exists/dist/src/index.js"(exports) {
    "use strict";
    var __importDefault = exports && exports.__importDefault || function(mod) {
      return mod && mod.__esModule ? mod : { "default": mod };
    };
    Object.defineProperty(exports, "__esModule", { value: true });
    var fs_1 = __require("fs");
    var debug_1 = __importDefault(require_src());
    var log = debug_1.default("@kwsites/file-exists");
    function check(path23, isFile, isDirectory) {
      log(`checking %s`, path23);
      try {
        const stat = fs_1.statSync(path23);
        if (stat.isFile() && isFile) {
          log(`[OK] path represents a file`);
          return true;
        }
        if (stat.isDirectory() && isDirectory) {
          log(`[OK] path represents a directory`);
          return true;
        }
        log(`[FAIL] path represents something other than a file or directory`);
        return false;
      } catch (e) {
        if (e.code === "ENOENT") {
          log(`[FAIL] path is not accessible: %o`, e);
          return false;
        }
        log(`[FATAL] %o`, e);
        throw e;
      }
    }
    function exists2(path23, type = exports.READABLE) {
      return check(path23, (type & exports.FILE) > 0, (type & exports.FOLDER) > 0);
    }
    exports.exists = exists2;
    exports.FILE = 1;
    exports.FOLDER = 2;
    exports.READABLE = exports.FILE + exports.FOLDER;
  }
});

// ../../node_modules/@kwsites/file-exists/dist/index.js
var require_dist = __commonJS({
  "../../node_modules/@kwsites/file-exists/dist/index.js"(exports) {
    "use strict";
    function __export3(m) {
      for (var p2 in m) if (!exports.hasOwnProperty(p2)) exports[p2] = m[p2];
    }
    Object.defineProperty(exports, "__esModule", { value: true });
    __export3(require_src2());
  }
});

// ../../node_modules/@kwsites/promise-deferred/dist/index.js
var require_dist2 = __commonJS({
  "../../node_modules/@kwsites/promise-deferred/dist/index.js"(exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.createDeferred = exports.deferred = void 0;
    function deferred2() {
      let done;
      let fail;
      let status = "pending";
      const promise = new Promise((_done, _fail) => {
        done = _done;
        fail = _fail;
      });
      return {
        promise,
        done(result) {
          if (status === "pending") {
            status = "resolved";
            done(result);
          }
        },
        fail(error) {
          if (status === "pending") {
            status = "rejected";
            fail(error);
          }
        },
        get fulfilled() {
          return status !== "pending";
        },
        get status() {
          return status;
        }
      };
    }
    exports.deferred = deferred2;
    exports.createDeferred = deferred2;
    exports.default = deferred2;
  }
});

// electron/main.ts
import { execFile as execFile4, execFileSync as execFileSync6, spawn as spawn5 } from "node:child_process";
import crypto6 from "node:crypto";
import fs19 from "node:fs";
import http2 from "node:http";
import https3 from "node:https";
import os6 from "node:os";
import path22 from "node:path";
import tls from "node:tls";
import { pathToFileURL as pathToFileURL3 } from "node:url";
import {
  app as app2,
  BrowserWindow as BrowserWindow2,
  clipboard,
  dialog,
  net as electronNet,
  globalShortcut,
  ipcMain,
  Menu as Menu2,
  nativeImage as nativeImage2,
  nativeTheme,
  Notification,
  powerMonitor,
  powerSaveBlocker,
  protocol,
  safeStorage,
  screen as screen2,
  session as session2,
  shell,
  systemPreferences
} from "electron";
import nodePty from "node-pty";

// electron/active-runtime-state.ts
function hasValidBootstrapMarker(marker, schemaVersion) {
  if (!marker || typeof marker !== "object") {
    return false;
  }
  if (marker.schemaVersion !== schemaVersion) {
    return false;
  }
  if (typeof marker.pinnedCommit !== "string" || marker.pinnedCommit.length < 7) {
    return false;
  }
  return true;
}
function classifyActiveRuntime(marker, schemaVersion, runtimeUsable) {
  const hasValidMarker = hasValidBootstrapMarker(marker, schemaVersion);
  if (!runtimeUsable) {
    return {
      hasValidMarker,
      shouldUseActiveRuntime: false,
      usabilityReason: "unusable"
    };
  }
  return {
    hasValidMarker,
    shouldUseActiveRuntime: true,
    usabilityReason: "usable"
  };
}

// electron/backend-child.ts
function stopBackendChild(child, deps) {
  if (!child || child.killed) {
    return;
  }
  const isWindows = deps.isWindows ?? process.platform === "win32";
  try {
    if (isWindows && Number.isInteger(child.pid)) {
      deps.forceKillProcessTree(child.pid);
    } else {
      child.kill("SIGTERM");
    }
  } catch {
  }
}

// electron/backend-command.ts
function dashboardFallbackArgs(args) {
  const i2 = args.indexOf("serve");
  if (i2 === -1) {
    return args.slice();
  }
  return [...args.slice(0, i2), "dashboard", "--no-open", ...args.slice(i2 + 1)];
}
function sourceDeclaresServe(dashboardPySource) {
  return /add_parser\(\s*["']serve["']/.test(String(dashboardPySource || ""));
}

// electron/backend-connection-state.ts
function createBackendConnectionState() {
  let generation = 0;
  let process3 = null;
  let promise = null;
  return {
    startAttempt() {
      return { generation, promise: null };
    },
    setPromise(attempt, nextPromise) {
      if (attempt.generation !== generation) {
        return false;
      }
      attempt.promise = nextPromise;
      promise = nextPromise;
      return true;
    },
    attachProcess(attempt, nextProcess) {
      if (attempt.generation !== generation) {
        return null;
      }
      process3 = nextProcess;
      return { generation, process: nextProcess };
    },
    clearForCurrentProcess(owner) {
      if (owner.generation !== generation || owner.process !== process3) {
        return false;
      }
      process3 = null;
      promise = null;
      return true;
    },
    clearPromiseForAttempt(attempt) {
      if (attempt.generation !== generation || promise !== null && attempt.promise !== promise) {
        return false;
      }
      promise = null;
      return true;
    },
    getProcess() {
      return process3;
    },
    getPromise() {
      return promise;
    },
    invalidate() {
      const currentProcess = process3;
      generation += 1;
      process3 = null;
      promise = null;
      return currentProcess;
    }
  };
}

// electron/backend-env.ts
import path from "node:path";
var POSIX_SANE_PATH_ENTRIES = Object.freeze([
  "/opt/homebrew/bin",
  "/opt/homebrew/sbin",
  "/usr/local/sbin",
  "/usr/local/bin",
  "/usr/sbin",
  "/usr/bin",
  "/sbin",
  "/bin"
]);
function delimiterForPlatform(platform = process.platform) {
  return platform === "win32" ? ";" : ":";
}
function pathModuleForPlatform(platform = process.platform) {
  return platform === "win32" ? path.win32 : path.posix;
}
function pathEnvKey(env2 = process.env, platform = process.platform) {
  if (platform !== "win32") {
    return "PATH";
  }
  return Object.keys(env2 || {}).find((key) => key.toUpperCase() === "PATH") || "PATH";
}
function currentPathValue(env2 = process.env, platform = process.platform) {
  const key = pathEnvKey(env2, platform);
  return env2?.[key] || "";
}
function appendUniquePathEntries(entries, { delimiter = path.delimiter } = {}) {
  const seen = /* @__PURE__ */ new Set();
  const ordered = [];
  for (const entry of entries) {
    if (!entry) {
      continue;
    }
    const parts = Array.isArray(entry) ? entry : String(entry).split(delimiter);
    for (const part of parts) {
      if (!part || seen.has(part)) {
        continue;
      }
      seen.add(part);
      ordered.push(part);
    }
  }
  return ordered.join(delimiter);
}
function hermesManagedNodePathEntries(hermesHome, { platform = process.platform, pathModule = pathModuleForPlatform(platform) } = {}) {
  if (!hermesHome) {
    return [];
  }
  const root = pathModule.join(hermesHome, "node");
  const bin = pathModule.join(root, "bin");
  return platform === "win32" ? [root, bin] : [bin, root];
}
function buildDesktopBackendPath({
  hermesHome,
  venvRoot,
  currentPath = "",
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform)
} = {}) {
  const delimiter = delimiterForPlatform(platform);
  const hermesNodeDirs = hermesManagedNodePathEntries(hermesHome, { platform, pathModule });
  const venvBin = venvRoot ? pathModule.join(venvRoot, platform === "win32" ? "Scripts" : "bin") : null;
  const saneEntries = platform === "win32" ? [] : POSIX_SANE_PATH_ENTRIES;
  return appendUniquePathEntries([hermesNodeDirs, venvBin, currentPath, saneEntries], { delimiter });
}
function normalizeHermesHomeRoot(hermesHome, { pathModule = pathModuleForPlatform(process.platform) } = {}) {
  if (!hermesHome) {
    return hermesHome;
  }
  const resolved = pathModule.resolve(String(hermesHome));
  const parent = pathModule.dirname(resolved);
  if (pathModule.basename(parent).toLowerCase() === "profiles") {
    return pathModule.dirname(parent);
  }
  return resolved;
}
function buildDesktopBackendEnv({
  hermesHome,
  pythonPathEntries = [],
  venvRoot,
  currentEnv = process.env,
  platform = process.platform,
  pathModule = pathModuleForPlatform(platform)
} = {}) {
  const delimiter = delimiterForPlatform(platform);
  const currentPythonPath = currentEnv?.PYTHONPATH || "";
  const key = pathEnvKey(currentEnv, platform);
  return {
    PYTHONPATH: appendUniquePathEntries([...pythonPathEntries, currentPythonPath], { delimiter }),
    // Force PEP 540 UTF-8 mode in the spawned Python backend so its stdio and
    // subprocess defaults are UTF-8 even on non-UTF-8 Windows locales (GBK,
    // cp1252, ...). hermes_bootstrap sets this inside the child too, but only
    // after import — anything emitted earlier (interpreter startup errors,
    // pre-bootstrap tracebacks) still decodes with the locale default without
    // this. User's explicit setting wins. Re-port of PR #56499 (echoriver89).
    PYTHONUTF8: currentEnv?.PYTHONUTF8 ?? "1",
    [key]: buildDesktopBackendPath({
      hermesHome,
      venvRoot,
      currentPath: currentPathValue(currentEnv, platform),
      platform,
      pathModule
    })
  };
}

// electron/backend-health.ts
var DEFAULT_BACKEND_READY_TIMEOUT_MS = 45e3;
var DEFAULT_BACKEND_READY_POLL_MS = 500;
var DEFAULT_HEALTH_PROBE_TIMEOUT_MS = 5e3;
var REMOTE_SESSION_EXPIRED_MESSAGE = 'Your remote gateway session has expired. Open Settings \u2192 Gateway and click "Sign in" again.';
function isMissingHealthEndpointError(error) {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return /^404:/.test(message) || message.includes("endpoint is likely missing");
}
function isAuthRejectionError(error) {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return /^40[13]:/.test(message);
}
function isGatedMissingHealthError(error) {
  const message = error instanceof Error ? error.message : String(error ?? "");
  return isAuthRejectionError(error) && message.includes("no_cookie");
}
function makeReauthRequiredError(detail) {
  const error = new Error(REMOTE_SESSION_EXPIRED_MESSAGE);
  error.needsOauthLogin = true;
  error.isReauthRequired = true;
  if (detail) {
    error.detail = detail;
  }
  return error;
}
function isReauthRequiredError(error) {
  return Boolean(error?.isReauthRequired);
}
function supersededError() {
  const error = new Error("SSH bootstrap was superseded by newer connection settings.");
  error.kind = "superseded";
  return error;
}
async function waitForHermesReady(baseUrl, options) {
  const timeoutMs = options.timeoutMs ?? DEFAULT_BACKEND_READY_TIMEOUT_MS;
  const pollMs = options.pollMs ?? DEFAULT_BACKEND_READY_POLL_MS;
  const healthProbeTimeoutMs = options.healthProbeTimeoutMs ?? DEFAULT_HEALTH_PROBE_TIMEOUT_MS;
  const now = options.now ?? Date.now;
  const signal = options.signal;
  const sleep2 = options.sleep ?? ((ms) => new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(supersededError());
      },
      { once: true }
    );
  }));
  const base = baseUrl.replace(/\/+$/, "");
  const deadline = now() + timeoutMs;
  const probeHealth = options.probeHealth ?? options.fetchPublicJson;
  const probeIsCredentialed = Boolean(options.probeIsCredentialed);
  let lastError = null;
  let useStatusFallback = false;
  while (now() < deadline) {
    if (signal?.aborted) {
      throw supersededError();
    }
    try {
      if (useStatusFallback) {
        await options.fetchJson(`${base}/api/status`, options.token);
      } else {
        await probeHealth(`${base}/api/health`, { timeoutMs: healthProbeTimeoutMs });
      }
      return;
    } catch (error) {
      lastError = error;
      if (probeIsCredentialed && isAuthRejectionError(error)) {
        throw makeReauthRequiredError(error instanceof Error ? error.message : String(error));
      }
      if (!useStatusFallback && (isMissingHealthEndpointError(error) || isGatedMissingHealthError(error))) {
        useStatusFallback = true;
        continue;
      }
      await sleep2(pollMs);
    }
  }
  const detail = lastError instanceof Error ? lastError.message : "timeout";
  throw new Error(`Hermes backend did not become ready: ${detail}`);
}

// electron/backend-probes.ts
import { execFileSync } from "node:child_process";
var DEFAULT_PROBE_TIMEOUT_MS = 15e3;
function resolveProbeTimeoutMs(env2 = process.env) {
  const raw = env2.HERMES_PROBE_TIMEOUT_MS;
  if (raw == null || raw === "") {
    return DEFAULT_PROBE_TIMEOUT_MS;
  }
  const n = Number.parseInt(String(raw), 10);
  if (!Number.isFinite(n) || n <= 0) {
    return DEFAULT_PROBE_TIMEOUT_MS;
  }
  return Math.min(n, 12e4);
}
var PROBE_TIMEOUT_MS = resolveProbeTimeoutMs();
function isTimeoutError(err) {
  if (!err || typeof err !== "object") {
    return false;
  }
  const e = err;
  if (e.killed === true) {
    return true;
  }
  if (e.code === "ETIMEDOUT") {
    return true;
  }
  if (e.signal === "SIGTERM") {
    return true;
  }
  return false;
}
function execProbeSync(command, args, options) {
  try {
    execFileSync(command, args, options);
  } catch (err) {
    if (!isTimeoutError(err)) {
      throw err;
    }
    execFileSync(command, args, options);
  }
}
function hermesRuntimeImportProbe() {
  return "import yaml; import dotenv; import hermes_cli.config";
}
function canImportHermesCli(pythonPath, opts = {}) {
  if (!pythonPath) {
    return false;
  }
  try {
    execProbeSync(pythonPath, ["-c", hermesRuntimeImportProbe()], {
      env: { ...process.env, ...opts.env || {} },
      stdio: "ignore",
      timeout: PROBE_TIMEOUT_MS,
      windowsHide: true
    });
    return true;
  } catch {
    return false;
  }
}
function shouldTrustHermesOverride(hermesOverride) {
  return typeof hermesOverride === "string" && hermesOverride.trim().length > 0;
}
function verifyHermesCli(hermesCommand, opts) {
  if (!hermesCommand) {
    return false;
  }
  try {
    execProbeSync(hermesCommand, ["--version"], {
      stdio: "ignore",
      timeout: PROBE_TIMEOUT_MS,
      shell: Boolean(opts?.shell),
      windowsHide: true
    });
    return true;
  } catch {
    return false;
  }
}

// electron/backend-ready.ts
import fs from "node:fs";
var _READY_RE = /^HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)/m;
var DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS = 9e4;
var MIN_PORT_ANNOUNCE_TIMEOUT_MS = 45e3;
function resolvePortAnnounceTimeoutMs(env2 = process.env) {
  const parsed = Number(env2.HERMES_DESKTOP_PORT_ANNOUNCE_TIMEOUT_MS);
  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.max(MIN_PORT_ANNOUNCE_TIMEOUT_MS, Math.round(parsed));
  }
  return DEFAULT_PORT_ANNOUNCE_TIMEOUT_MS;
}
function waitForDashboardPort(child, timeoutMs = resolvePortAnnounceTimeoutMs()) {
  return new Promise((resolve, reject) => {
    let buf = "";
    let done = false;
    function cleanup() {
      if (done) {
        return;
      }
      done = true;
      clearTimeout(timer);
      child.stdout.off("data", onData);
      child.off("exit", onExit);
      child.off("error", onError2);
    }
    function onData(chunk) {
      buf += chunk.toString();
      let nl;
      while ((nl = buf.indexOf("\n")) !== -1) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        const m = line.match(_READY_RE);
        if (m) {
          cleanup();
          resolve(parseInt(m[1], 10));
          return;
        }
      }
    }
    function onExit(code, signal) {
      cleanup();
      reject(new Error(`Hermes backend: exited before port announcement (${signal || code})`));
    }
    function onError2(err) {
      cleanup();
      reject(err);
    }
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error(`Timed out waiting for Hermes backend port announcement (${timeoutMs}ms)`));
    }, timeoutMs);
    child.stdout.on("data", onData);
    child.on("exit", onExit);
    child.on("error", onError2);
  });
}
function readDashboardReadyFile(readyFile) {
  if (!readyFile) {
    return null;
  }
  try {
    const parsed = JSON.parse(fs.readFileSync(readyFile, "utf8"));
    const port = Number(parsed?.port);
    return Number.isInteger(port) && port > 0 ? port : null;
  } catch {
    return null;
  }
}
function waitForDashboardReadyFile(readyFile, child, timeoutMs = resolvePortAnnounceTimeoutMs()) {
  return new Promise((resolve, reject) => {
    let done = false;
    let interval = null;
    function cleanup() {
      if (done) {
        return;
      }
      done = true;
      clearTimeout(timer);
      if (interval) {
        clearInterval(interval);
      }
      child.off("exit", onExit);
      child.off("error", onError2);
    }
    function check() {
      const port = readDashboardReadyFile(readyFile);
      if (port) {
        cleanup();
        resolve(port);
      }
    }
    function onExit(code, signal) {
      cleanup();
      reject(new Error(`Hermes backend: exited before port announcement (${signal || code})`));
    }
    function onError2(err) {
      cleanup();
      reject(err);
    }
    const timer = setTimeout(() => {
      cleanup();
      reject(new Error(`Timed out waiting for Hermes backend port announcement (${timeoutMs}ms)`));
    }, timeoutMs);
    child.on("exit", onExit);
    child.on("error", onError2);
    interval = setInterval(check, 50);
    if (typeof interval.unref === "function") {
      interval.unref();
    }
    check();
  });
}
function waitForDashboardPortAnnouncement(child, options = {}) {
  const timeoutMs = options.timeoutMs ?? resolvePortAnnounceTimeoutMs();
  if (options.readyFile) {
    return waitForDashboardReadyFile(options.readyFile, child, timeoutMs);
  }
  return waitForDashboardPort(child, timeoutMs);
}

// electron/backend-start-failure.ts
function shouldLatchBackendStartFailure(context) {
  return !context.attemptedRemote;
}
function shouldLatchRemoteReauthFailure(context) {
  return context.attemptedRemote && context.isReauth;
}

// electron/bootstrap-platform.ts
import fs2 from "node:fs";
function isWslEnvironment(env2 = process.env, platform = process.platform, kernelRelease = null) {
  if (platform !== "linux") {
    return false;
  }
  if (env2.WSL_DISTRO_NAME || env2.WSL_INTEROP) {
    return true;
  }
  try {
    const release = kernelRelease ?? fs2.readFileSync("/proc/sys/kernel/osrelease", "utf8");
    return /microsoft|wsl/i.test(release);
  } catch {
    return false;
  }
}
function isWindowsBinaryPathInWsl(filePath, options = {}) {
  const isWsl = options.isWsl ?? isWslEnvironment(options.env, options.platform);
  if (!isWsl) {
    return false;
  }
  const normalized = String(filePath || "").replace(/\\/g, "/").toLowerCase();
  return normalized.endsWith(".exe") || normalized.endsWith(".cmd") || normalized.endsWith(".bat") || normalized.endsWith(".ps1");
}
var GPU_OVERRIDE_ON = /* @__PURE__ */ new Set(["1", "true", "yes", "on"]);
var GPU_OVERRIDE_OFF = /* @__PURE__ */ new Set(["0", "false", "no", "off"]);
function detectRemoteDisplay(options = {}) {
  const env2 = options.env ?? process.env;
  const platform = options.platform ?? process.platform;
  const override = String(env2.HERMES_DESKTOP_DISABLE_GPU || "").trim().toLowerCase();
  if (GPU_OVERRIDE_ON.has(override)) {
    return "override (HERMES_DESKTOP_DISABLE_GPU)";
  }
  if (GPU_OVERRIDE_OFF.has(override)) {
    return null;
  }
  if (env2.SSH_CONNECTION || env2.SSH_CLIENT || env2.SSH_TTY) {
    return "ssh-session";
  }
  if (platform === "linux") {
    const display = String(env2.DISPLAY || "");
    if (display.includes(":") && display.split(":")[0]) {
      return `x11-forwarding (DISPLAY=${display})`;
    }
  }
  if (platform === "win32") {
    const sessionName = String(env2.SESSIONNAME || "");
    if (/^rdp-/i.test(sessionName)) {
      return `rdp (SESSIONNAME=${sessionName})`;
    }
  }
  return null;
}

// electron/bootstrap-repair-guard.ts
function decideBootstrapRepair(input) {
  const maxSoftAttempts = input.maxSoftAttempts ?? 3;
  const attempt = Math.max(1, Math.floor(input.attempt));
  const alive = Boolean(input.primaryBackendAlive);
  if (attempt > maxSoftAttempts) {
    return {
      hardReinstall: true,
      attempt,
      reason: `repair attempt ${attempt} exceeds soft-restart budget (${maxSoftAttempts}); escalating to hard reinstall`
    };
  }
  return {
    hardReinstall: false,
    attempt,
    reason: alive ? `repair attempt ${attempt}/${maxSoftAttempts}: primary backend process still alive (likely transient stall, see #74874); restarting only, skipping installer` : `repair attempt ${attempt}/${maxSoftAttempts}: primary backend process has exited; restarting before escalating to reinstall`
  };
}

// electron/bootstrap-runner.ts
import { execFileSync as execFileSync2, spawn } from "node:child_process";
import fs3 from "node:fs";
import fsp from "node:fs/promises";
import https from "node:https";
import path2 from "node:path";

// electron/windows-child-options.ts
function hiddenWindowsChildOptions(options = {}, isWindows = process.platform === "win32") {
  if (!isWindows || Object.prototype.hasOwnProperty.call(options, "windowsHide")) {
    return options;
  }
  return { ...options, windowsHide: true };
}

// electron/bootstrap-runner.ts
var IS_WINDOWS = process.platform === "win32";
var STAMP_COMMIT_RE = /^[0-9a-f]{7,40}$/i;
var FALLBACK_COMMIT_RE = /^0{7,40}$/;
var FALLBACK_BRANCH = "main";
function isPinnedCommit(commit) {
  return typeof commit === "string" && STAMP_COMMIT_RE.test(commit) && !FALLBACK_COMMIT_RE.test(commit);
}
function resolveCheckoutHead(activeRoot, opts = {}) {
  if (!activeRoot) {
    return null;
  }
  const run = opts.execGit || ((args, cwd) => execFileSync2("git", args, {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "ignore"],
    timeout: 15e3,
    ...hiddenWindowsChildOptions()
  }).trim());
  try {
    const sha = run(["-c", "windows.appendAtomically=false", "rev-parse", "HEAD"], activeRoot);
    return isPinnedCommit(sha) ? sha : null;
  } catch {
    return null;
  }
}
function readExistingPinnedCommit(activeRoot) {
  if (!activeRoot) {
    return null;
  }
  try {
    const raw = fs3.readFileSync(path2.join(activeRoot, ".hermes-bootstrap-complete"), "utf8");
    const parsed = JSON.parse(raw);
    return parsed && isPinnedCommit(parsed.pinnedCommit) ? parsed.pinnedCommit : null;
  } catch {
    return null;
  }
}
function resolveMarkerPinnedCommit(installStamp, activeRoot, opts = {}) {
  const resolveHead = opts.resolveHead || resolveCheckoutHead;
  if (installStamp && isPinnedCommit(installStamp.commit)) {
    return installStamp.commit;
  }
  const head = resolveHead(activeRoot);
  if (head) {
    return head;
  }
  return readExistingPinnedCommit(activeRoot);
}
function installRefForStamp(installStamp) {
  if (installStamp && isPinnedCommit(installStamp.commit)) {
    return {
      ref: installStamp.commit,
      cacheKey: installStamp.commit,
      pinned: true
    };
  }
  if (installStamp && typeof installStamp.commit === "string" && FALLBACK_COMMIT_RE.test(installStamp.commit)) {
    const ref = installStamp.branch || FALLBACK_BRANCH;
    return {
      ref,
      cacheKey: `fallback-${String(ref).replace(/[^0-9A-Za-z._-]/g, "_")}`,
      pinned: false
    };
  }
  return null;
}
function installScriptName() {
  return process.platform === "win32" ? "install.ps1" : "install.sh";
}
function installScriptKind() {
  return process.platform === "win32" ? "powershell" : "posix";
}
function resolveLocalInstallScript(sourceRepoRoot) {
  if (!sourceRepoRoot) {
    return null;
  }
  const candidate = path2.join(sourceRepoRoot, "scripts", installScriptName());
  try {
    fs3.accessSync(candidate, fs3.constants.R_OK);
    return candidate;
  } catch {
    return null;
  }
}
function bootstrapCacheDir(hermesHome) {
  return path2.join(hermesHome, "bootstrap-cache");
}
function installedAgentInstallScript(hermesHome) {
  if (!hermesHome) {
    return null;
  }
  const candidate = path2.join(hermesHome, "hermes-agent", "scripts", installScriptName());
  try {
    fs3.accessSync(candidate, fs3.constants.R_OK);
    return candidate;
  } catch {
    return null;
  }
}
function hasExistingGitCheckout(activeRoot) {
  if (!activeRoot) {
    return false;
  }
  try {
    return fs3.existsSync(path2.join(activeRoot, ".git"));
  } catch {
    return false;
  }
}
function cachedScriptPath(hermesHome, commit) {
  return path2.join(bootstrapCacheDir(hermesHome), `install-${commit}.${process.platform === "win32" ? "ps1" : "sh"}`);
}
function downloadInstallScript(ref, destPath) {
  const scriptName = installScriptName();
  const url = `https://raw.githubusercontent.com/NousResearch/hermes-agent/${ref}/scripts/${scriptName}`;
  return new Promise((resolve, reject) => {
    fs3.mkdirSync(path2.dirname(destPath), { recursive: true });
    const tmpPath = destPath + ".tmp";
    const out = fs3.createWriteStream(tmpPath);
    https.get(url, (res) => {
      if (res.statusCode === 301 || res.statusCode === 302) {
        out.close();
        fs3.unlinkSync(tmpPath);
        https.get(res.headers.location, (res2) => {
          if (res2.statusCode !== 200) {
            reject(
              new Error(
                `Failed to download ${scriptName}: HTTP ${res2.statusCode} from redirect ${res.headers.location}`
              )
            );
            return;
          }
          const out2 = fs3.createWriteStream(tmpPath);
          res2.pipe(out2);
          out2.on("finish", () => {
            out2.close();
            fs3.renameSync(tmpPath, destPath);
            resolve(destPath);
          });
          out2.on("error", reject);
        }).on("error", reject);
        return;
      }
      if (res.statusCode !== 200) {
        out.close();
        try {
          fs3.unlinkSync(tmpPath);
        } catch {
        }
        reject(new Error(`Failed to download ${scriptName}: HTTP ${res.statusCode} from ${url}`));
        return;
      }
      res.pipe(out);
      out.on("finish", () => {
        out.close();
        fs3.renameSync(tmpPath, destPath);
        resolve(destPath);
      });
      out.on("error", (err) => {
        try {
          fs3.unlinkSync(tmpPath);
        } catch {
        }
        reject(err);
      });
    }).on("error", (err) => {
      try {
        fs3.unlinkSync(tmpPath);
      } catch {
      }
      reject(err);
    });
  });
}
async function resolveInstallScript({
  installStamp,
  sourceRepoRoot,
  hermesHome,
  emit,
  _download = downloadInstallScript
}) {
  const localScript = resolveLocalInstallScript(sourceRepoRoot);
  if (localScript) {
    emit({ type: "log", line: `[bootstrap] using local ${installScriptName()} at ${localScript}` });
    return { path: localScript, source: "local", kind: installScriptKind() };
  }
  const installRef = installRefForStamp(installStamp);
  if (!installRef) {
    throw new Error(
      `Cannot resolve ${installScriptName()}: no SOURCE_REPO_ROOT and no install stamp. This packaged build was produced without a valid build-time stamp.`
    );
  }
  const cached = cachedScriptPath(hermesHome, installRef.cacheKey);
  const resolvedCommit = installRef.pinned ? installRef.ref : null;
  try {
    await fsp.access(cached, fs3.constants.R_OK);
    emit({
      type: "log",
      line: `[bootstrap] using cached ${installScriptName()} for ${installRef.ref.slice(0, 12)}`
    });
    return { path: cached, source: "cache", commit: resolvedCommit, kind: installScriptKind() };
  } catch {
  }
  emit({
    type: "log",
    line: `[bootstrap] fetching ${installScriptName()} for ${installRef.ref.slice(0, 12)} from GitHub` + (installRef.pinned ? "" : " (fallback, unpinned)")
  });
  try {
    await _download(installRef.ref, cached);
    emit({ type: "log", line: `[bootstrap] saved to ${cached}` });
    return { path: cached, source: "download", commit: resolvedCommit, kind: installScriptKind() };
  } catch (err) {
    const installed = installedAgentInstallScript(hermesHome);
    if (installed) {
      emit({
        type: "log",
        line: `[bootstrap] GitHub fetch failed (${err.message}); falling back to installed agent ${installScriptName()} at ${installed}`
      });
      try {
        fs3.mkdirSync(path2.dirname(cached), { recursive: true });
        fs3.copyFileSync(installed, cached);
        return { path: cached, source: "installed-agent", commit: resolvedCommit, kind: installScriptKind() };
      } catch {
        return { path: installed, source: "installed-agent", commit: resolvedCommit, kind: installScriptKind() };
      }
    }
    throw err;
  }
}
function powershellUnderRoot(root) {
  return path2.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
}
function resolveWindowsPowerShell() {
  for (const v of ["SystemRoot", "windir"]) {
    const root = process.env[v];
    if (root) {
      const candidate = powershellUnderRoot(root);
      try {
        if (fs3.statSync(candidate).isFile()) {
          return candidate;
        }
      } catch {
      }
    }
  }
  const pathDirs = (process.env.PATH || process.env.Path || "").split(path2.delimiter).filter(Boolean);
  for (const exe of ["powershell.exe", "pwsh.exe"]) {
    for (const dir of pathDirs) {
      const candidate = path2.join(dir, exe);
      try {
        if (fs3.statSync(candidate).isFile()) {
          return candidate;
        }
      } catch {
      }
    }
  }
  return "powershell.exe";
}
function spawnPowerShell(scriptPath, args, { emit, stageName, abortSignal, hermesHome } = {}) {
  return new Promise((resolve, reject) => {
    const ps = process.platform === "win32" ? resolveWindowsPowerShell() : "pwsh";
    const fullArgs = ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", scriptPath, ...args];
    const child = spawn(
      ps,
      fullArgs,
      hiddenWindowsChildOptions({
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          ...process.env,
          // Pass HERMES_HOME through so install.ps1 respects the caller's
          // choice rather than re-computing the default.
          HERMES_HOME: hermesHome || process.env.HERMES_HOME || ""
        }
      })
    );
    let stdout = "";
    let stderr = "";
    let killed = false;
    const onAbort = () => {
      killed = true;
      try {
        child.kill("SIGTERM");
      } catch {
      }
    };
    if (abortSignal) {
      if (abortSignal.aborted) {
        onAbort();
      } else {
        abortSignal.addEventListener("abort", onAbort, { once: true });
      }
    }
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    let stdoutBuf = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      stdoutBuf += chunk;
      let nl;
      while ((nl = stdoutBuf.indexOf("\n")) !== -1) {
        const line = stdoutBuf.slice(0, nl).replace(/\r$/, "");
        stdoutBuf = stdoutBuf.slice(nl + 1);
        if (line) {
          emit && emit({ type: "log", stage: stageName, line, stream: "stdout" });
        }
      }
    });
    let stderrBuf = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      stderrBuf += chunk;
      let nl;
      while ((nl = stderrBuf.indexOf("\n")) !== -1) {
        const line = stderrBuf.slice(0, nl).replace(/\r$/, "");
        stderrBuf = stderrBuf.slice(nl + 1);
        if (line) {
          emit && emit({ type: "log", stage: stageName, line, stream: "stderr" });
        }
      }
    });
    child.on("error", (err) => {
      if (abortSignal) {
        abortSignal.removeEventListener("abort", onAbort);
      }
      reject(err);
    });
    child.on("close", (code, signal) => {
      if (abortSignal) {
        abortSignal.removeEventListener("abort", onAbort);
      }
      if (stdoutBuf) {
        emit && emit({ type: "log", stage: stageName, line: stdoutBuf, stream: "stdout" });
      }
      if (stderrBuf) {
        emit && emit({ type: "log", stage: stageName, line: stderrBuf, stream: "stderr" });
      }
      resolve({ stdout, stderr, code, signal, killed });
    });
  });
}
function spawnBash(scriptPath, args, { emit, stageName, abortSignal, hermesHome } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn("bash", [scriptPath, ...args], {
      stdio: ["ignore", "pipe", "pipe"],
      env: {
        ...process.env,
        HERMES_HOME: hermesHome || process.env.HERMES_HOME || ""
      }
    });
    let stdout = "";
    let stderr = "";
    let killed = false;
    const onAbort = () => {
      killed = true;
      try {
        child.kill("SIGTERM");
      } catch {
      }
    };
    if (abortSignal) {
      if (abortSignal.aborted) {
        onAbort();
      } else {
        abortSignal.addEventListener("abort", onAbort, { once: true });
      }
    }
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    let stdoutBuf = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      stdoutBuf += chunk;
      let nl;
      while ((nl = stdoutBuf.indexOf("\n")) !== -1) {
        const line = stdoutBuf.slice(0, nl).replace(/\r$/, "");
        stdoutBuf = stdoutBuf.slice(nl + 1);
        if (line) {
          emit && emit({ type: "log", stage: stageName, line, stream: "stdout" });
        }
      }
    });
    let stderrBuf = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk;
      stderrBuf += chunk;
      let nl;
      while ((nl = stderrBuf.indexOf("\n")) !== -1) {
        const line = stderrBuf.slice(0, nl).replace(/\r$/, "");
        stderrBuf = stderrBuf.slice(nl + 1);
        if (line) {
          emit && emit({ type: "log", stage: stageName, line, stream: "stderr" });
        }
      }
    });
    child.on("error", (err) => {
      if (abortSignal) {
        abortSignal.removeEventListener("abort", onAbort);
      }
      reject(err);
    });
    child.on("close", (code, signal) => {
      if (abortSignal) {
        abortSignal.removeEventListener("abort", onAbort);
      }
      if (stdoutBuf) {
        emit && emit({ type: "log", stage: stageName, line: stdoutBuf, stream: "stdout" });
      }
      if (stderrBuf) {
        emit && emit({ type: "log", stage: stageName, line: stderrBuf, stream: "stderr" });
      }
      resolve({ stdout, stderr, code, signal, killed });
    });
  });
}
function buildPinArgs(installStamp, { pinCommit = true } = {}) {
  const args = [];
  if (pinCommit && installStamp && isPinnedCommit(installStamp.commit)) {
    args.push("-Commit", installStamp.commit);
  }
  if (installStamp && installStamp.branch) {
    args.push("-Branch", installStamp.branch);
  }
  return args;
}
function buildPosixPinArgs({ installStamp, activeRoot, hermesHome, pinCommit = true }) {
  const args = ["--dir", activeRoot, "--hermes-home", hermesHome];
  if (installStamp && installStamp.branch) {
    args.push("--branch", installStamp.branch);
  }
  if (pinCommit && installStamp && isPinnedCommit(installStamp.commit)) {
    args.push("--commit", installStamp.commit);
  }
  return args;
}
async function fetchManifest({ scriptPath, installerKind, emit, hermesHome, activeRoot, installStamp, pinCommit }) {
  const isPosix = installerKind === "posix";
  const args = isPosix ? ["--manifest", ...buildPosixPinArgs({ installStamp, activeRoot, hermesHome, pinCommit })] : ["-Manifest", ...buildPinArgs(installStamp, { pinCommit })];
  const result = await (isPosix ? spawnBash : spawnPowerShell)(scriptPath, args, {
    emit,
    stageName: "__manifest__",
    hermesHome
  });
  if (result.code !== 0) {
    throw new Error(
      `${isPosix ? "install.sh --manifest" : "install.ps1 -Manifest"} failed: exit ${result.code}
${result.stderr || result.stdout}`
    );
  }
  const lines = result.stdout.split(/\r?\n/).filter(Boolean);
  for (let i2 = lines.length - 1; i2 >= 0; i2--) {
    try {
      const parsed = JSON.parse(lines[i2]);
      if (parsed && Array.isArray(parsed.stages)) {
        return parsed;
      }
    } catch {
    }
  }
  throw new Error(
    `${isPosix ? "install.sh --manifest" : "install.ps1 -Manifest"} produced no parseable JSON payload
${result.stdout}`
  );
}
function parseStageResult(stdout) {
  const lines = stdout.split(/\r?\n/).filter(Boolean);
  for (let i2 = lines.length - 1; i2 >= 0; i2--) {
    try {
      const parsed = JSON.parse(lines[i2]);
      if (parsed && typeof parsed.ok === "boolean" && typeof parsed.stage === "string") {
        return parsed;
      }
    } catch {
    }
  }
  return null;
}
async function runStage({
  scriptPath,
  installerKind,
  stage,
  emit,
  hermesHome,
  activeRoot,
  abortSignal,
  installStamp,
  pinCommit
}) {
  const startedAt = Date.now();
  emit({ type: "stage", name: stage.name, state: "running" });
  const isPosix = installerKind === "posix";
  const args = isPosix ? [
    "--stage",
    stage.name,
    "--non-interactive",
    "--json",
    ...buildPosixPinArgs({ installStamp, activeRoot, hermesHome, pinCommit })
  ] : ["-Stage", stage.name, "-NonInteractive", "-Json", ...buildPinArgs(installStamp, { pinCommit })];
  const result = await (isPosix ? spawnBash : spawnPowerShell)(scriptPath, args, {
    emit,
    stageName: stage.name,
    abortSignal,
    hermesHome
  });
  const durationMs = Date.now() - startedAt;
  if (result.killed) {
    const ev2 = { type: "stage", name: stage.name, state: "failed", durationMs, error: "cancelled by user" };
    emit(ev2);
    return ev2;
  }
  const json = parseStageResult(result.stdout);
  if (!json) {
    const ev2 = {
      type: "stage",
      name: stage.name,
      state: "failed",
      durationMs,
      error: `${isPosix ? "install.sh --stage" : "install.ps1 -Stage"} ${stage.name} produced no JSON result frame (exit=${result.code})`,
      json: null
    };
    emit(ev2);
    return ev2;
  }
  if (json.ok && json.skipped) {
    const ev2 = { type: "stage", name: stage.name, state: "skipped", durationMs, json };
    emit(ev2);
    return ev2;
  }
  if (json.ok) {
    const ev2 = { type: "stage", name: stage.name, state: "succeeded", durationMs, json };
    emit(ev2);
    return ev2;
  }
  const ev = {
    type: "stage",
    name: stage.name,
    state: "failed",
    durationMs,
    json,
    error: json.reason || `exit code ${result.code}`
  };
  emit(ev);
  return ev;
}
function openRunLog(logRoot) {
  fs3.mkdirSync(logRoot, { recursive: true });
  const ts = (/* @__PURE__ */ new Date()).toISOString().replace(/[:.]/g, "-");
  const logPath = path2.join(logRoot, `bootstrap-${ts}.log`);
  const stream = fs3.createWriteStream(logPath, { flags: "a" });
  return { path: logPath, stream };
}
async function runBootstrap(opts) {
  const {
    installStamp,
    activeRoot,
    sourceRepoRoot,
    hermesHome,
    logRoot,
    onEvent,
    abortSignal,
    writeMarker
    // callback to write the bootstrap-complete marker; main.ts provides
  } = opts;
  if (abortSignal && abortSignal.aborted) {
    if (typeof onEvent === "function") {
      try {
        onEvent({ type: "failed", error: "bootstrap cancelled by user" });
      } catch {
      }
    }
    return { ok: false, cancelled: true };
  }
  const runLog = openRunLog(logRoot || path2.join(hermesHome, "logs"));
  const emit = (ev) => {
    try {
      runLog.stream.write(JSON.stringify(ev) + "\n");
    } catch {
    }
    try {
      if (typeof onEvent === "function") {
        onEvent(ev);
      }
    } catch (err) {
      runLog.stream.write(`emit error: ${err && err.message}
`);
    }
  };
  emit({
    type: "log",
    line: `[bootstrap] starting at ${(/* @__PURE__ */ new Date()).toISOString()}; activeRoot=${activeRoot}; stamp=${installStamp ? installStamp.commit.slice(0, 12) : "<none>"}; runLog=${runLog.path}`
  });
  try {
    const existingCheckout = hasExistingGitCheckout(activeRoot);
    const pinCommit = !existingCheckout;
    if (existingCheckout && installStamp && installStamp.commit) {
      emit({
        type: "log",
        line: `[bootstrap] existing checkout detected at ${activeRoot}; not pinning to packaged install stamp ${installStamp.commit.slice(0, 12)}`
      });
    }
    const scriptInfo = await resolveInstallScript({ installStamp, sourceRepoRoot, hermesHome, emit });
    const installerKind = scriptInfo.kind || "powershell";
    const manifest = await fetchManifest({
      scriptPath: scriptInfo.path,
      installerKind,
      emit,
      hermesHome,
      activeRoot,
      installStamp,
      pinCommit
    });
    emit({
      type: "manifest",
      stages: manifest.stages,
      protocolVersion: manifest.protocol_version || manifest.protocolVersion || null
    });
    for (const stage of manifest.stages) {
      if (abortSignal && abortSignal.aborted) {
        emit({ type: "failed", error: "bootstrap cancelled by user" });
        return { ok: false, cancelled: true };
      }
      const ev = await runStage({
        scriptPath: scriptInfo.path,
        installerKind,
        stage,
        emit,
        hermesHome,
        activeRoot,
        abortSignal,
        installStamp,
        pinCommit
      });
      if (ev.state === "failed") {
        emit({ type: "failed", stage: stage.name, error: ev.error || "stage failed" });
        return { ok: false, failedStage: stage.name, error: ev.error };
      }
    }
    const pinnedCommit = resolveMarkerPinnedCommit(installStamp, activeRoot);
    if (!pinnedCommit) {
      emit({
        type: "log",
        line: "[bootstrap] WARNING: could not resolve a real pinnedCommit for the bootstrap-complete marker; subsequent launches may re-run bootstrap"
      });
    } else if (installStamp && !isPinnedCommit(installStamp.commit)) {
      emit({
        type: "log",
        line: `[bootstrap] fallback stamp resolved marker pin to ${pinnedCommit.slice(0, 12)} from checkout`
      });
    }
    const markerPayload = {
      pinnedCommit,
      pinnedBranch: installStamp ? installStamp.branch : null
    };
    const marker = typeof writeMarker === "function" ? writeMarker(markerPayload) : markerPayload;
    emit({ type: "complete", marker });
    return { ok: true, marker };
  } catch (err) {
    emit({ type: "failed", error: err.message || String(err) });
    return { ok: false, error: err.message || String(err) };
  } finally {
    try {
      runLog.stream.end();
    } catch {
    }
  }
}

// electron/connection-apply.ts
async function applyConnectionChange({
  cancelAndWait,
  isPrimary,
  rehomePrimary = null,
  scope,
  sendApplied,
  stopPool,
  teardownPrimary,
  teardownSsh
}) {
  await cancelAndWait(scope);
  await teardownSsh(scope);
  if (!isPrimary) {
    stopPool(scope);
    return;
  }
  if (rehomePrimary) {
    await rehomePrimary();
    return;
  }
  await teardownPrimary();
  sendApplied();
}
async function resolveTerminalConnection(getTarget, ensureBackend2) {
  let target2 = getTarget();
  if (target2 !== "pending") {
    return target2;
  }
  await ensureBackend2();
  target2 = getTarget();
  if (target2 === "pending") {
    throw new Error("Remote connection is not ready yet. Try again in a moment.");
  }
  return target2;
}

// electron/connection-config.ts
var AT_COOKIE_VARIANTS = ["__Host-hermes_session_at", "__Secure-hermes_session_at", "hermes_session_at"];
var RT_COOKIE_VARIANTS = ["__Host-hermes_session_rt", "__Secure-hermes_session_rt", "hermes_session_rt"];
var PRIVY_SESSION_COOKIE_VARIANTS = ["__Host-privy-token", "__Secure-privy-token", "privy-token", "privy-session"];
var RESERVED_REMOTE_PROFILES = /* @__PURE__ */ new Set(["hermes", "test", "tmp", "root", "sudo"]);
function normalizeRemoteBaseUrl(rawUrl) {
  let value = String(rawUrl || "").trim();
  if (!value) {
    throw new Error("Remote gateway URL is required.");
  }
  if (!/^[a-z][a-z0-9+.-]*:\/\//i.test(value)) {
    value = `http://${value}`;
  }
  let parsed;
  try {
    parsed = new URL(value);
  } catch (error) {
    throw new Error(`Remote gateway URL is not valid: ${error.message}`);
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(`Remote gateway URL must be http:// or https://, got ${parsed.protocol}`);
  }
  parsed.hash = "";
  parsed.search = "";
  parsed.pathname = parsed.pathname.replace(/\/+$/, "");
  return parsed.toString().replace(/\/+$/, "");
}
function buildGatewayWsUrl(baseUrl, token) {
  const parsed = new URL(baseUrl);
  const wsScheme = parsed.protocol === "https:" ? "wss" : "ws";
  const prefix = parsed.pathname.replace(/\/+$/, "");
  return `${wsScheme}://${parsed.host}${prefix}/api/ws?token=${encodeURIComponent(token)}`;
}
function buildGatewayWsUrlWithTicket(baseUrl, ticket) {
  const parsed = new URL(baseUrl);
  const wsScheme = parsed.protocol === "https:" ? "wss" : "ws";
  const prefix = parsed.pathname.replace(/\/+$/, "");
  return `${wsScheme}://${parsed.host}${prefix}/api/ws?ticket=${encodeURIComponent(ticket)}`;
}
function isGatewayAuthRejection(error) {
  if (error && typeof error === "object" && error.needsOauthLogin === true) {
    return true;
  }
  const statusCode = Number(error && typeof error === "object" ? error.statusCode : NaN);
  return statusCode === 401 || statusCode === 403;
}
function gatewayTicketFailure(error, authMessage, transportMessage) {
  const needsOauthLogin = isGatewayAuthRejection(error);
  const err = new Error(needsOauthLogin ? authMessage : transportMessage);
  if (needsOauthLogin) {
    ;
    err.needsOauthLogin = true;
  }
  err.cause = error;
  return err;
}
async function gatewayWsUrlIpcResult(resolveWsUrl) {
  try {
    return { ok: true, wsUrl: await resolveWsUrl() };
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : String(error),
      ...isGatewayAuthRejection(error) ? { needsOauthLogin: true } : {},
      ok: false
    };
  }
}
async function resolveTestWsUrl(baseUrl, authMode, token, deps = {}) {
  if (authMode === "oauth") {
    const mintTicket = deps.mintTicket;
    if (typeof mintTicket !== "function") {
      throw new Error("resolveTestWsUrl: a mintTicket function is required in OAuth mode.");
    }
    let ticket;
    try {
      ticket = await mintTicket(baseUrl);
    } catch (error) {
      throw gatewayTicketFailure(
        error,
        "Reached the gateway over HTTP, but the OAuth session was rejected while minting a WebSocket ticket. Open Settings \u2192 Gateway and sign in again.",
        "Reached the gateway over HTTP, but could not mint a WebSocket ticket. Check the remote gateway connection and try again."
      );
    }
    return buildGatewayWsUrlWithTicket(baseUrl, ticket);
  }
  if (!token) {
    return null;
  }
  return buildGatewayWsUrl(baseUrl, token);
}
function connectionScopeKey(profile) {
  return String(profile ?? "").trim() || null;
}
function normAuthMode(mode) {
  return mode === "oauth" ? "oauth" : "token";
}
function modeIsRemoteLike(mode) {
  return mode === "remote" || mode === "cloud";
}
function normalizeSshConfig(entry) {
  if (!entry || typeof entry !== "object" || entry.mode !== "ssh") {
    return null;
  }
  let host = String(entry.host || "").trim();
  if (!host) {
    return null;
  }
  let parsedUser;
  let parsedPort;
  const at = host.indexOf("@");
  if (at > 0) {
    parsedUser = host.slice(0, at);
    host = host.slice(at + 1);
  }
  const bracketed = /^\[([^\]]+)](?::(\d+))?$/.exec(host);
  if (bracketed) {
    host = bracketed[1];
    if (bracketed[2]) {
      parsedPort = Number(bracketed[2]);
    }
  } else if ((host.match(/:/g) || []).length === 1) {
    const [name, rawPort] = host.split(":");
    if (/^\d+$/.test(rawPort)) {
      host = name;
      parsedPort = Number(rawPort);
    }
  }
  if (!host) {
    return null;
  }
  const out = { mode: "ssh", host };
  const user = String(entry.user || "").trim() || parsedUser || "";
  if (user) {
    out.user = user;
  }
  const rawExplicitPort = String(entry.port ?? "").trim();
  const explicitPort = /^\d+$/.test(rawExplicitPort) ? Number(rawExplicitPort) : null;
  const port = explicitPort ?? parsedPort;
  if (Number.isInteger(port) && port > 0 && port <= 65535 && port !== 22) {
    out.port = port;
  }
  const keyPath = String(entry.keyPath || "").trim();
  if (keyPath) {
    out.keyPath = keyPath;
  }
  const remoteHermesPath = String(entry.remoteHermesPath || "").trim();
  if (remoteHermesPath) {
    out.remoteHermesPath = remoteHermesPath;
  }
  const remoteProfile = String(entry.remoteProfile || "").trim();
  if (/^[a-z0-9][a-z0-9_-]{0,63}$/.test(remoteProfile) && !RESERVED_REMOTE_PROFILES.has(remoteProfile)) {
    out.remoteProfile = remoteProfile;
  }
  return out;
}
function profileSshOverride(config, profile) {
  const key = connectionScopeKey(profile);
  const entry = key ? config?.profiles?.[key] : null;
  return normalizeSshConfig(entry);
}
function savedProfileSsh(config, profile) {
  const key = connectionScopeKey(profile);
  const entry = key ? config?.profiles?.[key] : null;
  if (!entry || entry.mode !== "local") {
    return null;
  }
  return normalizeSshConfig(entry.savedSsh);
}
function profileHasRemoteConnection(config, profile) {
  return Boolean(profileRemoteOverride(config, profile) || profileSshOverride(config, profile));
}
function localProfileEntry(existing) {
  const ssh = normalizeSshConfig(existing) || normalizeSshConfig(existing?.savedSsh);
  return ssh ? { mode: "local", savedSsh: ssh } : null;
}
function hostLabelFromBaseUrl(baseUrl) {
  const raw = String(baseUrl || "").trim();
  if (!raw) {
    return null;
  }
  try {
    const parsed = new URL(raw);
    if (!parsed.hostname) {
      return null;
    }
    return parsed.port && parsed.port !== "80" && parsed.port !== "443" ? `${parsed.hostname}:${parsed.port}` : parsed.hostname;
  } catch {
    return null;
  }
}
function profileRemoteOverride(config, profile) {
  const key = connectionScopeKey(profile);
  const entry = key ? config?.profiles?.[key] : null;
  if (!entry || typeof entry !== "object" || !modeIsRemoteLike(entry.mode)) {
    return null;
  }
  const url = String(entry.url || "").trim();
  if (!url) {
    return null;
  }
  return { url, authMode: normAuthMode(entry.authMode), token: entry.token };
}
function resolveProfileBackendRoute(profile, opts = {}) {
  const scopedProfile = connectionScopeKey(profile);
  const primaryProfile = connectionScopeKey(opts.primaryProfile) || "default";
  if (!scopedProfile || scopedProfile === primaryProfile) {
    return { backend: "primary", descriptorProfile: null, scopePath: false };
  }
  if (opts.profileRemoteOverride) {
    return { backend: "pool", descriptorProfile: null, scopePath: false };
  }
  if (opts.globalRemote) {
    return { backend: "primary", descriptorProfile: scopedProfile, scopePath: true };
  }
  return { backend: "pool", descriptorProfile: null, scopePath: false };
}
function pathWithGlobalRemoteProfile(path23, profile, opts = {}) {
  const scopedProfile = connectionScopeKey(profile);
  if (!resolveProfileBackendRoute(profile, opts).scopePath) {
    return path23;
  }
  const rawPath = String(path23 || "");
  if (!rawPath) {
    return path23;
  }
  let parsed;
  try {
    parsed = new URL(rawPath, "http://hermes.local");
  } catch {
    return path23;
  }
  if (parsed.searchParams.has("profile")) {
    return path23;
  }
  parsed.searchParams.set("profile", scopedProfile);
  return `${parsed.pathname}${parsed.search}${parsed.hash}`;
}
function tokenPreview(value) {
  const raw = String(value || "");
  if (!raw) {
    return null;
  }
  return raw.length <= 8 ? "set" : `...${raw.slice(-6)}`;
}
function authModeFromStatus(statusBody) {
  return statusBody && statusBody.auth_required ? "oauth" : "token";
}
function resolveAuthMode(inputAuthMode, existingAuthMode) {
  if (inputAuthMode === "oauth") {
    return "oauth";
  }
  if (inputAuthMode === "token") {
    return "token";
  }
  if (existingAuthMode === "oauth") {
    return "oauth";
  }
  return "token";
}
function cookiesHaveSession(cookies) {
  if (!Array.isArray(cookies)) {
    return false;
  }
  return cookies.some((c3) => c3 && AT_COOKIE_VARIANTS.includes(c3.name) && c3.value);
}
function cookiesHaveLiveSession(cookies) {
  if (!Array.isArray(cookies)) {
    return false;
  }
  return cookies.some((c3) => c3 && c3.value && (AT_COOKIE_VARIANTS.includes(c3.name) || RT_COOKIE_VARIANTS.includes(c3.name)));
}
function cookiesHavePrivySession(cookies) {
  if (!Array.isArray(cookies)) {
    return false;
  }
  return cookies.some((c3) => c3 && c3.value && PRIVY_SESSION_COOKIE_VARIANTS.includes(c3.name));
}

// electron/crash-forensics.ts
function describeCrashReason(reason) {
  if (reason instanceof Error) {
    return reason.stack || reason.message || reason.name || "Error";
  }
  if (typeof reason === "string") {
    return reason;
  }
  try {
    return JSON.stringify(reason) ?? String(reason);
  } catch {
    return String(reason);
  }
}
function installCrashForensics({ flush, log, target: target2 = process }) {
  const record = (label) => (reason) => {
    log(`[main] ${label}: ${describeCrashReason(reason)}`);
    flush();
  };
  target2.on("uncaughtException", record("Uncaught exception"));
  target2.on("unhandledRejection", record("Unhandled rejection"));
}

// electron/dashboard-token.ts
var DEFAULT_TOKEN_FETCH_TIMEOUT_MS = 3e3;
async function fetchPublicText(url, options = {}) {
  const { protocol: protocol2 } = new URL(url);
  if (protocol2 !== "http:" && protocol2 !== "https:") {
    throw new Error(`Unsupported Hermes backend URL protocol: ${protocol2}`);
  }
  const timeoutMs = options.timeoutMs ?? DEFAULT_TOKEN_FETCH_TIMEOUT_MS;
  const res = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) }).catch((error) => {
    if (error.name === "TimeoutError") {
      throw new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`);
    }
    throw error;
  });
  const text = await res.text();
  if (!res.ok) {
    throw new Error(`${res.status}: ${text || res.statusText}`);
  }
  return text;
}
function extractInjectedDashboardToken(html) {
  const match = /window\.__HERMES_SESSION_TOKEN__\s*=\s*("(?:\\.|[^"\\])*")/.exec(String(html || ""));
  if (!match) {
    return null;
  }
  try {
    return JSON.parse(match[1]);
  } catch {
    return null;
  }
}
function dashboardIndexUrl(baseUrl) {
  return `${String(baseUrl || "").replace(/\/+$/, "")}/`;
}
async function resolveServedDashboardToken(baseUrl, fallbackToken, options = {}) {
  const fetchText = options.fetchText || fetchPublicText;
  const html = await fetchText(dashboardIndexUrl(baseUrl), {
    timeoutMs: options.timeoutMs ?? DEFAULT_TOKEN_FETCH_TIMEOUT_MS
  });
  const servedToken = extractInjectedDashboardToken(html);
  if (servedToken && servedToken !== fallbackToken && typeof options.rememberLog === "function") {
    options.rememberLog("[boot] dashboard served a different session token; using served token for WebSocket auth");
  }
  return servedToken || fallbackToken;
}
function isForeignBackendToken({ servedToken, spawnToken, childAlive }) {
  return Boolean(servedToken) && servedToken !== spawnToken && !childAlive;
}
async function adoptServedDashboardToken(baseUrl, spawnToken, { childAlive, label = "Hermes backend", ...options }) {
  const servedToken = await resolveServedDashboardToken(baseUrl, spawnToken, options).catch((error) => {
    options.rememberLog?.(`[boot] could not read served dashboard token (${label}): ${error.message}`);
    return spawnToken;
  });
  if (isForeignBackendToken({ servedToken, spawnToken, childAlive: childAlive() })) {
    throw new Error(
      `${label} exited and ${dashboardIndexUrl(baseUrl)} is served by a process we did not spawn; refusing its session token.`
    );
  }
  return servedToken;
}

// electron/desktop-installation.ts
import crypto from "node:crypto";
import fs4 from "node:fs";
import path3 from "node:path";
var INSTALLATION_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
function parseInstallationId(raw) {
  try {
    const value = JSON.parse(String(raw || ""))?.installationId;
    return INSTALLATION_ID_RE.test(value) ? value.toLowerCase() : "";
  } catch {
    return "";
  }
}
function readInstallationId(filePath) {
  try {
    const stat = fs4.lstatSync(filePath);
    if (!stat.isFile() || stat.isSymbolicLink()) {
      return "";
    }
    if (typeof process.getuid === "function" && stat.uid !== process.getuid()) {
      return "";
    }
    if (process.platform !== "win32" && (stat.mode & 511) !== 384) {
      fs4.chmodSync(filePath, 384);
    }
    return parseInstallationId(fs4.readFileSync(filePath, "utf8"));
  } catch {
    return "";
  }
}
function waitForRepair() {
  const buffer = new SharedArrayBuffer(4);
  Atomics.wait(new Int32Array(buffer), 0, 0, 25);
}
function loadOrCreateInstallationId(filePath, randomUUID = crypto.randomUUID) {
  const existing = readInstallationId(filePath);
  if (existing) {
    return existing;
  }
  fs4.mkdirSync(path3.dirname(filePath), { recursive: true });
  const installationId = randomUUID().toLowerCase();
  if (!INSTALLATION_ID_RE.test(installationId)) {
    throw new Error("Could not generate a valid desktop installation ID.");
  }
  const repairPath = `${filePath}.repair.lock`;
  for (let attempt = 0; attempt < 40; attempt++) {
    let repairFd;
    try {
      repairFd = fs4.openSync(repairPath, "wx", 384);
    } catch (error) {
      if (error?.code !== "EEXIST") {
        throw error;
      }
      const winner = readInstallationId(filePath);
      if (winner) {
        return winner;
      }
      waitForRepair();
      continue;
    }
    try {
      const winner = readInstallationId(filePath);
      if (winner) {
        return winner;
      }
      try {
        const stat = fs4.lstatSync(filePath);
        if (!stat.isFile() && !stat.isSymbolicLink()) {
          throw new Error("Desktop installation ID path is not a regular file.");
        }
        if (!stat.isSymbolicLink() && typeof process.getuid === "function" && stat.uid !== process.getuid()) {
          throw new Error("Desktop installation ID is owned by another user.");
        }
        fs4.unlinkSync(filePath);
      } catch (error) {
        if (error?.code !== "ENOENT") {
          throw error;
        }
      }
      fs4.writeFileSync(filePath, JSON.stringify({ installationId }), { encoding: "utf8", flag: "wx", mode: 384 });
      return installationId;
    } finally {
      if (repairFd !== void 0) {
        fs4.closeSync(repairFd);
      }
      try {
        fs4.unlinkSync(repairPath);
      } catch {
      }
    }
  }
  throw new Error("Could not repair the desktop installation ID.");
}
function sshOwnershipId(installationId, scope) {
  if (!INSTALLATION_ID_RE.test(String(installationId || ""))) {
    throw new Error("Desktop installation ID is invalid.");
  }
  return crypto.createHash("sha256").update(`${installationId}\0${String(scope || "")}`).digest("hex").slice(0, 32);
}

// electron/desktop-uninstall.ts
import path4 from "node:path";
var UNINSTALL_MODES = ["gui", "lite", "full"];
function uninstallArgsForMode(mode) {
  if (!UNINSTALL_MODES.includes(mode)) {
    throw new Error(`Unknown uninstall mode: ${mode}`);
  }
  return ["-m", "hermes_cli.uninstall", "--mode", mode];
}
function modeRemovesAgent(mode) {
  return mode === "lite" || mode === "full";
}
function modeRemovesUserData(mode) {
  return mode === "full";
}
function resolveRemovableAppPath(execPath, platform, env2 = {}) {
  const exe = String(execPath || "");
  if (!exe) {
    return null;
  }
  const p2 = platform === "win32" ? path4.win32 : path4.posix;
  if (platform === "darwin") {
    const macOsDir = p2.dirname(exe);
    const contents = p2.dirname(macOsDir);
    const appBundle = p2.dirname(contents);
    if (appBundle.endsWith(".app")) {
      return appBundle;
    }
    return null;
  }
  if (platform === "win32") {
    const dir2 = p2.dirname(exe);
    if (/[\\/]Hermes$/i.test(dir2) || /[\\/]hermes-desktop$/i.test(dir2)) {
      return dir2;
    }
    return null;
  }
  if (env2.APPIMAGE) {
    return env2.APPIMAGE;
  }
  const dir = p2.dirname(exe);
  if (/-unpacked$/.test(dir)) {
    return dir;
  }
  return null;
}
function shouldRemoveAppBundle(isPackaged, appPath) {
  return Boolean(isPackaged) && Boolean(appPath);
}
function buildPosixCleanupScript({ desktopPid, pythonExe, pythonPath, agentRoot, uninstallArgs, appPath, hermesHome }) {
  const q2 = (s) => `'${String(s).replace(/'/g, `'\\''`)}'`;
  const lines = [
    "#!/bin/bash",
    "set -u",
    "# Wait (up to ~30s) for the desktop process to exit so the venv python",
    "# and the app bundle are no longer in use.",
    `pid=${Number(desktopPid) || 0}`,
    'if [ "$pid" -gt 0 ]; then',
    "  for _ in $(seq 1 60); do",
    '    kill -0 "$pid" 2>/dev/null || break',
    "    sleep 0.5",
    "  done",
    "fi",
    `export HERMES_HOME=${q2(hermesHome)}`
  ];
  if (pythonPath) {
    lines.push(`export PYTHONPATH=${q2(pythonPath)}\${PYTHONPATH:+:$PYTHONPATH}`);
  }
  lines.push(`cd ${q2(agentRoot)} 2>/dev/null || true`, `${q2(pythonExe)} ${uninstallArgs.map(q2).join(" ")} || true`);
  if (appPath) {
    lines.push(`rm -rf ${q2(appPath)} || true`);
  }
  lines.push('rm -f "$0" 2>/dev/null || true');
  lines.push("");
  return lines.join("\n");
}
function buildWindowsCleanupScript({
  desktopPid,
  pythonExe,
  pythonPath,
  agentRoot,
  uninstallArgs,
  appPath,
  hermesHome
}) {
  const pid = Number(desktopPid) || 0;
  const q2 = (s) => `"${String(s).replace(/"/g, "")}"`;
  const lines = [
    "@echo off",
    "setlocal enableextensions",
    `set "HERMES_HOME=${String(hermesHome).replace(/"/g, "")}"`,
    `set "PID=${pid}"`
  ];
  if (pythonPath) {
    lines.push(`set "PYTHONPATH=${String(pythonPath).replace(/"/g, "")};%PYTHONPATH%"`);
  }
  lines.push(
    "set /a waited=0",
    ":waitloop",
    'rem /FI "PID eq %PID%" is an EXACT filter \u2014 tasklist outputs the one task',
    'rem row for that PID, or "INFO: No tasks..." otherwise. /NH drops the',
    "rem header; findstr matches the PID as a whole space-delimited token so",
    "rem PID 99 cannot match 990 (the substring trap of a bare `find`).",
    'tasklist /NH /FI "PID eq %PID%" 2>nul | findstr /r /c:" %PID% " >nul',
    "if %ERRORLEVEL% neq 0 goto waited_done",
    "set /a waited+=1",
    "if %waited% geq 60 goto waited_done",
    "timeout /t 1 /nobreak >nul",
    "goto waitloop",
    ":waited_done",
    `cd /d ${q2(agentRoot)}`,
    `${q2(pythonExe)} ${uninstallArgs.map(q2).join(" ")}`
  );
  if (appPath) {
    lines.push(
      "set /a tries=0",
      ":rmloop",
      `if not exist ${q2(appPath)} goto rmdone`,
      `rmdir /s /q ${q2(appPath)} >nul 2>&1`,
      `if not exist ${q2(appPath)} goto rmdone`,
      "set /a tries+=1",
      "if %tries% geq 10 goto rmdone",
      "timeout /t 1 /nobreak >nul",
      "goto rmloop",
      ":rmdone"
    );
  }
  lines.push('del "%~f0"');
  lines.push("");
  return lines.join("\r\n");
}

// electron/dev-cdp.ts
var DEFAULT_PORT = 9222;
var MIN_PORT = 1024;
var MAX_PORT = 65535;
var OPT_OUT = /* @__PURE__ */ new Set(["0", "off", "false", "no"]);
function resolveDevCdpPort({ env: env2, isPackaged, devServer }) {
  if (isPackaged) {
    return { port: null, reason: "packaged" };
  }
  if (!devServer) {
    return { port: null, reason: "no-dev-server" };
  }
  const requested = (env2.HERMES_DESKTOP_CDP_PORT ?? "").trim();
  if (!requested) {
    return { port: DEFAULT_PORT, reason: null };
  }
  if (OPT_OUT.has(requested.toLowerCase())) {
    return { port: null, reason: "opted-out" };
  }
  const port = Number(requested);
  if (!Number.isInteger(port) || port < MIN_PORT || port > MAX_PORT) {
    return { port: null, reason: "invalid-port" };
  }
  return { port, reason: null };
}
function describeDevCdpDecision(decision) {
  switch (decision.reason) {
    case null:
      return null;
    case "invalid-port":
      return `HERMES_DESKTOP_CDP_PORT is not a valid port (expected an integer ${MIN_PORT}-${MAX_PORT}, or "off"); renderer debugging is disabled.`;
    case "opted-out":
      return "renderer debugging disabled by HERMES_DESKTOP_CDP_PORT.";
    // Packaged and dist-run builds are closed by design — the common case, not
    // worth a line of startup noise.
    case "packaged":
    case "no-dev-server":
      return null;
  }
}

// electron/embed-referer.ts
import { session } from "electron";
var EMBED_SESSION_PARTITION = "persist:hermes-embed";
var EMBED_REFERER = "https://www.youtube.com/";
var YOUTUBE_REFERER_HOST_RE = /(^|\.)(youtube\.com|youtube-nocookie\.com|googlevideo\.com|ytimg\.com|youtubei\.googleapis\.com)$/i;
function installEmbedRefererForSession(embedSession) {
  if (!embedSession) {
    return;
  }
  embedSession.webRequest.onBeforeSendHeaders((details, callback) => {
    let host = "";
    try {
      host = new URL(details.url).hostname;
    } catch {
      host = "";
    }
    if (!YOUTUBE_REFERER_HOST_RE.test(host)) {
      callback({ requestHeaders: details.requestHeaders });
      return;
    }
    const headers = { ...details.requestHeaders };
    if (!headers.Referer && !headers.referer) {
      headers.Referer = EMBED_REFERER;
    }
    callback({ requestHeaders: headers });
  });
}
function installEmbedReferer() {
  try {
    installEmbedRefererForSession(session.fromPartition(EMBED_SESSION_PARTITION));
  } catch {
  }
}

// electron/event-dedupe.ts
var DEDUPE_INTERVAL_MS = 1e3;
function createEventDeduper(intervalMs = DEDUPE_INTERVAL_MS) {
  const lastSeenAt = /* @__PURE__ */ new Map();
  return function isDuplicate(key, now = Date.now()) {
    for (const [k2, at] of lastSeenAt) {
      if (now - at >= intervalMs) {
        lastSeenAt.delete(k2);
      }
    }
    if (lastSeenAt.has(key)) {
      return true;
    }
    lastSeenAt.set(key, now);
    return false;
  };
}

// electron/find-git-bash.ts
import path5 from "node:path";
function findGitBash(opts) {
  const { isWindows, env: env2, fileExists: fileExists2, findOnPath: findOnPath2 } = opts;
  if (!isWindows) {
    return findOnPath2 ? findOnPath2("bash") : null;
  }
  const gitBashPath = env2.HERMES_GIT_BASH_PATH;
  if (gitBashPath && fileExists2(gitBashPath)) {
    return gitBashPath;
  }
  const localAppData = env2.LOCALAPPDATA || "";
  const candidates = [];
  const joinWin = path5.win32.join;
  if (localAppData) {
    candidates.push(joinWin(localAppData, "hermes", "git", "bin", "bash.exe"));
    candidates.push(joinWin(localAppData, "hermes", "git", "usr", "bin", "bash.exe"));
  }
  candidates.push(joinWin(env2["ProgramFiles"] || "C:\\Program Files", "Git", "bin", "bash.exe"));
  candidates.push(joinWin(env2["ProgramFiles(x86)"] || "C:\\Program Files (x86)", "Git", "bin", "bash.exe"));
  if (localAppData) {
    candidates.push(joinWin(localAppData, "Programs", "Git", "bin", "bash.exe"));
  }
  for (const candidate of candidates) {
    if (fileExists2(candidate)) {
      return candidate;
    }
  }
  if (findOnPath2) {
    const onPath = findOnPath2("bash");
    if (onPath) {
      return onPath;
    }
  }
  return null;
}

// electron/find-in-page.ts
function formatFoundInPage(result) {
  return {
    activeMatchOrdinal: Number(result?.activeMatchOrdinal ?? 0),
    count: Number(result?.matches ?? 0)
  };
}
function performFind(webContents, query, options) {
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  const opts = options && typeof options === "object" ? options : {};
  webContents.findInPage(String(query ?? ""), {
    forward: opts.forward !== false,
    findNext: Boolean(opts.findNext)
  });
}
function stopFind(webContents, action = "clearSelection") {
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  webContents.stopFindInPage(action);
}
function installFoundInPageForwarder(webContents) {
  if (!webContents || webContents.isDestroyed()) {
    return () => {
    };
  }
  const handler = (_event, result) => {
    if (webContents.isDestroyed()) {
      return;
    }
    webContents.send("hermes:found-in-page", formatFoundInPage(result));
  };
  webContents.on("found-in-page", handler);
  return () => {
    webContents.off("found-in-page", handler);
  };
}

// electron/first-run-setup-gate.ts
function createFirstRunSetupGate({
  hideChoice,
  log,
  onStuck,
  promptChoice,
  stuckAfterMs = 12e4
} = {}) {
  let localBootstrapConfirmed = false;
  let waiter = null;
  let stuckTimer = null;
  const clearStuckTimer = () => {
    if (stuckTimer) {
      clearTimeout(stuckTimer);
      stuckTimer = null;
    }
  };
  const armStuckTimer = (backend) => {
    clearStuckTimer();
    if (!Number.isFinite(stuckAfterMs) || stuckAfterMs <= 0 || typeof log !== "function") {
      return;
    }
    stuckTimer = setTimeout(() => {
      onStuck?.(backend, stuckAfterMs);
      log(
        `[bootstrap] still waiting for first-run setup choice after ${Math.round(stuckAfterMs / 1e3)}s (platform=${backend?.platform || "unknown"})`
      );
    }, stuckAfterMs);
    if (typeof stuckTimer.unref === "function") {
      stuckTimer.unref();
    }
  };
  const shouldGate = (backend) => Boolean(backend && backend.kind === "bootstrap-needed" && !localBootstrapConfirmed);
  const wait = async (backend) => {
    if (!shouldGate(backend)) {
      return "continue-local";
    }
    if (waiter) {
      return waiter.promise;
    }
    promptChoice?.(backend);
    armStuckTimer(backend);
    let resolveWaiter = () => {
    };
    const promise = new Promise((resolve) => {
      resolveWaiter = resolve;
    });
    waiter = { promise, resolve: resolveWaiter };
    return promise;
  };
  const settleWaiter = (decision) => {
    clearStuckTimer();
    if (!waiter) {
      return false;
    }
    const activeWaiter = waiter;
    waiter = null;
    activeWaiter.resolve(decision);
    return true;
  };
  const continueLocal = () => {
    localBootstrapConfirmed = true;
    settleWaiter("continue-local");
  };
  const resetForRetry = () => {
    settleWaiter("reset");
  };
  const resetForRepair = () => {
    resetForRetry();
    localBootstrapConfirmed = false;
  };
  const abandonForRemoteApply = () => {
    const resumedWaiter = settleWaiter("remote-applied");
    if (!resumedWaiter) {
      return false;
    }
    localBootstrapConfirmed = false;
    hideChoice?.();
    return true;
  };
  const isLocalBootstrapConfirmed = () => localBootstrapConfirmed;
  const hasWaiter = () => Boolean(waiter);
  return {
    abandonForRemoteApply,
    continueLocal,
    hasWaiter,
    isLocalBootstrapConfirmed,
    resetForRepair,
    resetForRetry,
    shouldGate,
    wait
  };
}

// electron/fs-read-dir.ts
import fs7 from "node:fs";
import path7 from "node:path";

// electron/hardening.ts
import fs5 from "node:fs";
import os from "node:os";
import path6 from "node:path";
import { fileURLToPath } from "node:url";
var DEFAULT_FETCH_TIMEOUT_MS = 15e3;
var DATA_URL_READ_DEFAULT_MAX_MB = 16;
var DATA_URL_READ_MIN_MAX_MB = 1;
var DATA_URL_READ_MAX_MAX_MB = 4096;
var ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES = 256 * 1024 * 1024;
var TEXT_PREVIEW_SOURCE_MAX_BYTES = 64 * 1024 * 1024;
function clampDataUrlReadMaxMb(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return DATA_URL_READ_DEFAULT_MAX_MB;
  }
  return Math.min(DATA_URL_READ_MAX_MAX_MB, Math.max(DATA_URL_READ_MIN_MAX_MB, Math.round(parsed)));
}
function dataUrlReadMaxBytesFromMb(maxMb) {
  return clampDataUrlReadMaxMb(maxMb) * 1024 * 1024;
}
var SAFE_ENV_SUFFIXES = /* @__PURE__ */ new Set(["dist", "example", "sample", "template"]);
var SENSITIVE_EXTENSIONS = /* @__PURE__ */ new Set([".kdbx", ".p12", ".pem", ".pfx"]);
function resolveTimeoutMs(timeoutMs, fallbackMs = DEFAULT_FETCH_TIMEOUT_MS) {
  const fallback = Number.isFinite(fallbackMs) && Number(fallbackMs) > 0 ? Math.round(Number(fallbackMs)) : DEFAULT_FETCH_TIMEOUT_MS;
  const parsed = Number(timeoutMs);
  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.round(parsed);
  }
  return fallback;
}
function encryptDesktopSecret(value, safeStorageApi) {
  const raw = String(value || "");
  if (!raw) {
    return null;
  }
  let encryptionAvailable = false;
  try {
    encryptionAvailable = Boolean(safeStorageApi?.isEncryptionAvailable?.());
  } catch {
    encryptionAvailable = false;
  }
  if (!encryptionAvailable) {
    throw new Error(
      "Secure token storage is unavailable, so Hermes Desktop cannot save remote gateway tokens. Set HERMES_DESKTOP_REMOTE_URL and HERMES_DESKTOP_REMOTE_TOKEN in your environment, or enable OS keychain access and try again."
    );
  }
  try {
    return {
      encoding: "safeStorage",
      value: safeStorageApi.encryptString(raw).toString("base64")
    };
  } catch (error) {
    const detail = error instanceof Error && error.message ? ` (${error.message})` : "";
    throw new Error(
      `Failed to encrypt the remote gateway token for secure storage${detail}. Set HERMES_DESKTOP_REMOTE_URL and HERMES_DESKTOP_REMOTE_TOKEN in your environment as a fallback.`
    );
  }
}
function sensitiveFileBlockReason(filePath) {
  const normalized = String(filePath || "").replace(/\\/g, "/").toLowerCase();
  const basename = path6.basename(normalized);
  const ext = path6.extname(basename);
  if (!basename) {
    return null;
  }
  if (normalized.includes("/.ssh/")) {
    return "SSH key/config files are blocked.";
  }
  if (normalized.includes("/.gnupg/")) {
    return "GPG key material is blocked.";
  }
  if (normalized.endsWith("/.aws/credentials")) {
    return "AWS credential files are blocked.";
  }
  if (basename === ".env") {
    return ".env files are blocked because they commonly contain secrets.";
  }
  if (basename.startsWith(".env.")) {
    const suffix = basename.slice(".env.".length);
    if (!SAFE_ENV_SUFFIXES.has(suffix)) {
      return `${basename} is blocked because it appears to contain environment secrets.`;
    }
  }
  if (/^id_(rsa|dsa|ecdsa|ed25519)(?:\..+)?$/.test(basename) && !basename.endsWith(".pub")) {
    return "SSH private key files are blocked.";
  }
  if (SENSITIVE_EXTENSIONS.has(ext)) {
    return `${ext} key/certificate files are blocked.`;
  }
  if (basename === ".npmrc" || basename === ".netrc" || basename === ".pypirc") {
    return `${basename} is blocked because it may include auth credentials.`;
  }
  return null;
}
function ipcPathError(code, message) {
  const error = new Error(message);
  error.code = code;
  return error;
}
function rejectUnsafePathSyntax(filePath, purpose = "File read") {
  if (typeof filePath !== "string") {
    throw ipcPathError("invalid-path", `${purpose} failed: file path is required.`);
  }
  const raw = filePath.trim();
  if (!raw) {
    throw ipcPathError("invalid-path", `${purpose} failed: file path is required.`);
  }
  if (raw.includes("\0")) {
    throw ipcPathError("invalid-path", `${purpose} failed: file path is invalid.`);
  }
  const normalized = raw.replace(/\\/g, "/").toLowerCase();
  if (normalized.startsWith("//?/") || normalized.startsWith("//./") || normalized.startsWith("globalroot/device/") || normalized.includes("/globalroot/device/")) {
    throw ipcPathError("device-path", `${purpose} blocked: Windows device paths are not allowed.`);
  }
  return raw;
}
function resolveRequestedPathForIpc(filePath, options = {}) {
  const purpose = String(options.purpose || "File read");
  let raw = rejectUnsafePathSyntax(filePath, purpose);
  if (raw === "~" || raw.startsWith("~/") || raw.startsWith("~\\")) {
    raw = path6.join(os.homedir(), raw.slice(1));
  }
  if (/^file:/i.test(raw)) {
    let resolvedPath2;
    try {
      const parsed = new URL(raw);
      if (parsed.protocol !== "file:") {
        throw new Error("not a file URL");
      }
      resolvedPath2 = fileURLToPath(parsed);
    } catch {
      throw ipcPathError("invalid-path", `${purpose} failed: file URL is invalid.`);
    }
    rejectUnsafePathSyntax(resolvedPath2, purpose);
    return path6.resolve(resolvedPath2);
  }
  const baseInput = typeof options.baseDir === "string" && options.baseDir.trim() ? options.baseDir : process.cwd();
  const safeBaseInput = rejectUnsafePathSyntax(baseInput, purpose);
  const resolvedBase = path6.resolve(safeBaseInput);
  rejectUnsafePathSyntax(resolvedBase, purpose);
  const resolvedPath = path6.resolve(resolvedBase, raw);
  rejectUnsafePathSyntax(resolvedPath, purpose);
  return resolvedPath;
}
async function statForIpc(fsImpl, resolvedPath, purpose, typeLabel) {
  try {
    return await fsImpl.promises.stat(resolvedPath);
  } catch (error) {
    const code = error && typeof error === "object" ? error.code : "";
    if (code === "ENOENT" || code === "ENOTDIR") {
      throw ipcPathError(code || "ENOENT", `${purpose} failed: ${typeLabel} does not exist.`);
    }
    throw ipcPathError(
      code || "read-error",
      `${purpose} failed: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}
async function realpathForIpc(fsImpl, resolvedPath, purpose) {
  if (typeof fsImpl.promises.realpath !== "function") {
    return resolvedPath;
  }
  try {
    const realPath = await fsImpl.promises.realpath(resolvedPath);
    rejectUnsafePathSyntax(realPath, purpose);
    return realPath;
  } catch (error) {
    const code = error && typeof error === "object" ? error.code : "";
    throw ipcPathError(
      code || "read-error",
      `${purpose} failed: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}
function rejectSensitiveFilePath(filePath, purpose) {
  const blockReason = sensitiveFileBlockReason(filePath);
  if (blockReason) {
    throw ipcPathError("sensitive-file", `${purpose} blocked for sensitive file: ${blockReason}`);
  }
}
async function resolveDirectoryForIpc(dirPath, options = {}) {
  const purpose = String(options.purpose || "Directory read");
  const fsImpl = options.fs || fs5;
  const resolvedPath = resolveRequestedPathForIpc(dirPath, { baseDir: options.baseDir, purpose });
  const stat = await statForIpc(fsImpl, resolvedPath, purpose, "directory");
  if (!stat.isDirectory()) {
    throw ipcPathError("ENOTDIR", `${purpose} failed: path is not a directory.`);
  }
  const realPath = await realpathForIpc(fsImpl, resolvedPath, purpose);
  return { realPath, resolvedPath, stat };
}
async function resolveReadableFileForIpc(filePath, options = {}) {
  const purpose = String(options.purpose || "File read");
  const fsImpl = options.fs || fs5;
  const resolvedPath = resolveRequestedPathForIpc(filePath, { baseDir: options.baseDir, purpose });
  if (options.blockSensitive !== false) {
    rejectSensitiveFilePath(resolvedPath, purpose);
  }
  const stat = await statForIpc(fsImpl, resolvedPath, purpose, "file");
  if (stat.isDirectory()) {
    throw ipcPathError("EISDIR", `${purpose} failed: path points to a directory.`);
  }
  if (!stat.isFile()) {
    throw ipcPathError("EINVAL", `${purpose} failed: only regular files can be read.`);
  }
  const realPath = await realpathForIpc(fsImpl, resolvedPath, purpose);
  if (options.blockSensitive !== false) {
    rejectSensitiveFilePath(realPath, purpose);
  }
  const maxBytes = Number.isFinite(options.maxBytes) && Number(options.maxBytes) > 0 ? Number(options.maxBytes) : null;
  if (maxBytes && stat.size > maxBytes) {
    throw ipcPathError("EFBIG", `${purpose} failed: file is too large (${stat.size} bytes; limit ${maxBytes} bytes).`);
  }
  try {
    await fsImpl.promises.access(resolvedPath, fs5.constants.R_OK);
  } catch {
    throw ipcPathError("EACCES", `${purpose} failed: file is not readable.`);
  }
  return { realPath, resolvedPath, stat };
}
async function readFileDataUrlForIpc(filePath, options) {
  const fsImpl = options.fs || fs5;
  const { resolvedPath } = await resolveReadableFileForIpc(filePath, options);
  const data = await fsImpl.promises.readFile(resolvedPath);
  return `data:${options.mimeType};base64,${data.toString("base64")}`;
}

// electron/wsl-path-bridge.ts
import { execFileSync as execFileSync3 } from "node:child_process";
import fs6 from "node:fs";
var IS_WINDOWS2 = process.platform === "win32";
var WIN_DRIVE_RE = /^([A-Za-z]):[\\/]/;
var WSL_MOUNT_RE = /^\/mnt\/([a-z])(?:\/(.*))?$/i;
var cachedDistro = null;
var cachedUncBase = null;
function parseDefaultDistro(raw) {
  return String(raw || "").replace(/\0/g, "").split(/\r?\n/).map((line) => line.replace(/^\*?\s*/, "").trim()).find(Boolean) || null;
}
function resolveDefaultWslDistro() {
  if (cachedDistro) {
    return cachedDistro;
  }
  if (!IS_WINDOWS2) {
    cachedDistro = "Ubuntu";
    return cachedDistro;
  }
  try {
    const out = execFileSync3("wsl.exe", ["-l", "-q"], {
      encoding: "utf8",
      env: { ...process.env, WSL_UTF8: "1" },
      timeout: 2e3,
      windowsHide: true
    });
    cachedDistro = parseDefaultDistro(out) || "Ubuntu";
  } catch {
    cachedDistro = "Ubuntu";
  }
  return cachedDistro;
}
function wslUncBase(distro) {
  if (cachedUncBase) {
    return cachedUncBase;
  }
  const modern = `\\\\wsl.localhost\\${distro}`;
  const legacy = `\\\\wsl$\\${distro}`;
  try {
    if (!fs6.existsSync(modern) && fs6.existsSync(legacy)) {
      cachedUncBase = legacy;
      return cachedUncBase;
    }
  } catch {
  }
  cachedUncBase = modern;
  return cachedUncBase;
}
function wslPosixToWindowsAccessible(posixPath, distro = resolveDefaultWslDistro()) {
  const value = String(posixPath || "").trim();
  const normalized = value.replace(/\\/g, "/");
  if (!normalized.startsWith("/")) {
    return value;
  }
  const mount = normalized.match(WSL_MOUNT_RE);
  if (mount) {
    const tail = (mount[2] || "").replace(/\//g, "\\");
    return tail ? `${mount[1].toUpperCase()}:\\${tail}` : `${mount[1].toUpperCase()}:\\`;
  }
  const relative = normalized.replace(/^\/+/, "").replace(/\//g, "\\");
  return `${wslUncBase(distro)}\\${relative}`;
}
function resolvePickerDefaultPath(defaultPath, distro = resolveDefaultWslDistro()) {
  if (!defaultPath) {
    return void 0;
  }
  const value = String(defaultPath).trim();
  return value.startsWith("/") && !WIN_DRIVE_RE.test(value) ? wslPosixToWindowsAccessible(value, distro) : defaultPath;
}
function resolveLocalReadPath(dirPath, distro = resolveDefaultWslDistro()) {
  const value = String(dirPath || "").trim();
  return IS_WINDOWS2 && value.startsWith("/") && !WIN_DRIVE_RE.test(value) ? wslPosixToWindowsAccessible(value, distro) : value;
}

// electron/fs-read-dir.ts
var FS_READDIR_STAT_CONCURRENCY = 16;
var FS_READDIR_HIDDEN = /* @__PURE__ */ new Set([
  ".git",
  ".hg",
  ".svn",
  ".cache",
  ".next",
  ".turbo",
  ".venv",
  "__pycache__",
  "build",
  "dist",
  "node_modules",
  "target",
  "venv"
]);
function direntIsDirectory(dirent) {
  return typeof dirent.isDirectory === "function" && dirent.isDirectory();
}
function direntIsFile(dirent) {
  return typeof dirent.isFile === "function" && dirent.isFile();
}
function direntIsSymbolicLink(dirent) {
  return typeof dirent.isSymbolicLink === "function" && dirent.isSymbolicLink();
}
function shouldStatDirent(dirent) {
  if (direntIsDirectory(dirent)) {
    return false;
  }
  return direntIsSymbolicLink(dirent) || !direntIsFile(dirent);
}
async function entryForDirent(dirent, resolved, fsImpl) {
  const fullPath = path7.join(resolved, dirent.name);
  let isDirectory = direntIsDirectory(dirent);
  if (!isDirectory && shouldStatDirent(dirent)) {
    try {
      isDirectory = (await fsImpl.promises.stat(fullPath)).isDirectory();
    } catch {
      isDirectory = false;
    }
  }
  return { name: dirent.name, path: fullPath, isDirectory };
}
async function mapWithStatConcurrency(items, mapper) {
  const results = new Array(items.length);
  let nextIndex = 0;
  async function runWorker() {
    while (nextIndex < items.length) {
      const index = nextIndex;
      nextIndex += 1;
      results[index] = await mapper(items[index]);
    }
  }
  const workerCount = Math.min(FS_READDIR_STAT_CONCURRENCY, items.length);
  const workers = Array.from({ length: workerCount }, () => runWorker());
  await Promise.all(workers);
  return results;
}
async function readDirForIpc(dirPath, options = {}) {
  const fsImpl = options.fs || fs7;
  let resolved;
  const readPath = resolveLocalReadPath(String(dirPath ?? ""));
  try {
    ;
    ({ resolvedPath: resolved } = await resolveDirectoryForIpc(readPath, {
      fs: fsImpl,
      purpose: "Directory read"
    }));
  } catch (error) {
    return { entries: [], error: error?.code || "read-error" };
  }
  try {
    const dirents = await fsImpl.promises.readdir(resolved, { withFileTypes: true });
    const visibleDirents = dirents.filter((dirent) => !FS_READDIR_HIDDEN.has(dirent.name));
    const entries = await mapWithStatConcurrency(visibleDirents, (dirent) => entryForDirent(dirent, resolved, fsImpl));
    entries.sort((a, b2) => Number(b2.isDirectory) - Number(a.isDirectory) || a.name.localeCompare(b2.name));
    return { entries };
  } catch (error) {
    return { entries: [], error: error?.code || "read-error" };
  }
}

// electron/gateway-ws-probe.ts
var DEFAULT_CONNECT_TIMEOUT_MS = 1e4;
var DEFAULT_READY_GRACE_MS = 750;
function probeGatewayWebSocket(wsUrl, options = {}) {
  const WebSocketImpl = options.WebSocketImpl;
  const connectTimeoutMs = options.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS;
  const readyGraceMs = options.readyGraceMs ?? DEFAULT_READY_GRACE_MS;
  if (typeof WebSocketImpl !== "function") {
    return Promise.resolve({
      ok: false,
      reason: "WebSocket is not available in this runtime."
    });
  }
  return new Promise((resolve) => {
    let settled = false;
    let opened = false;
    let connectTimer = null;
    let graceTimer = null;
    let socket;
    const clearTimers = () => {
      if (connectTimer !== null) {
        clearTimeout(connectTimer);
        connectTimer = null;
      }
      if (graceTimer !== null) {
        clearTimeout(graceTimer);
        graceTimer = null;
      }
    };
    const finish = (result) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimers();
      try {
        socket?.close?.();
      } catch {
      }
      resolve(result);
    };
    try {
      socket = new WebSocketImpl(wsUrl);
    } catch (error) {
      finish({
        ok: false,
        reason: error instanceof Error ? error.message : String(error)
      });
      return;
    }
    const onOpen = () => {
      if (settled) {
        return;
      }
      opened = true;
      graceTimer = setTimeout(() => {
        finish({ ok: true });
      }, readyGraceMs);
    };
    const onMessage = () => {
      finish({ ok: true });
    };
    const onError2 = (event) => {
      finish({
        ok: false,
        reason: extractErrorReason(event) || "WebSocket connection failed."
      });
    };
    const onClose = (event) => {
      if (settled) {
        return;
      }
      if (opened) {
        finish({
          ok: false,
          reason: closeReason(event, "The gateway accepted the connection then closed it (credential rejected?).")
        });
        return;
      }
      finish({
        ok: false,
        reason: closeReason(event, "The gateway closed the WebSocket before it opened.")
      });
    };
    addListener(socket, "open", onOpen);
    addListener(socket, "message", onMessage);
    addListener(socket, "error", onError2);
    addListener(socket, "close", onClose);
    if (connectTimeoutMs > 0) {
      connectTimer = setTimeout(() => {
        finish({
          ok: false,
          reason: `Timed out after ${connectTimeoutMs}ms waiting for the WebSocket to open.`
        });
      }, connectTimeoutMs);
    }
  });
}
function addListener(socket, type, handler) {
  if (typeof socket.addEventListener === "function") {
    socket.addEventListener(type, handler);
    return;
  }
  if (typeof socket.on === "function") {
    socket.on(type, handler);
  }
}
function extractErrorReason(event) {
  if (!event) {
    return "";
  }
  if (event instanceof Error) {
    return event.message;
  }
  const err = event.error || event.message;
  if (err instanceof Error) {
    return err.message;
  }
  if (typeof err === "string") {
    return err;
  }
  return "";
}
function closeReason(event, fallback) {
  const code = event && typeof event.code === "number" ? event.code : null;
  const reason = event && typeof event.reason === "string" ? event.reason.trim() : "";
  if (code && reason) {
    return `${fallback} (code ${code}: ${reason})`;
  }
  if (code) {
    return `${fallback} (code ${code})`;
  }
  if (reason) {
    return `${fallback} (${reason})`;
  }
  return fallback;
}

// electron/git-repo-scan.ts
import fs8 from "node:fs";
import os2 from "node:os";
import path8 from "node:path";
var fsp2 = fs8.promises;
var DEFAULT_MAX_DEPTH = 3;
var MAX_CONCURRENCY = 32;
var JUNK_DIRS = /* @__PURE__ */ new Set(["Applications", "Library", "node_modules", "site-packages", "vendor", "venv"]);
function pathApiFor(platform) {
  return platform === "win32" ? path8.win32 : path8.posix;
}
function normalizeRepoScanPath(rawPath, options = {}) {
  const platform = options.platform ?? process.platform;
  const homeDir = options.homeDir ?? os2.homedir();
  const pathApi = pathApiFor(platform);
  const raw = String(rawPath ?? "").trim();
  if (!raw) {
    return null;
  }
  let expanded = raw;
  if (raw === "~") {
    expanded = homeDir;
  } else if (raw.startsWith("~/") || raw.startsWith("~\\")) {
    expanded = pathApi.join(homeDir, raw.slice(2));
  }
  const absolute = pathApi.isAbsolute(expanded) ? expanded : pathApi.resolve(homeDir, expanded);
  const value = pathApi.normalize(absolute);
  const key = platform === "win32" ? value.toLocaleLowerCase("en-US") : value;
  return { key, value };
}
function repoScanPathIsWithin(candidate, parent, options = {}) {
  const platform = options.platform ?? process.platform;
  const pathApi = pathApiFor(platform);
  const candidatePath = normalizeRepoScanPath(candidate, options);
  const parentPath = normalizeRepoScanPath(parent, options);
  if (!candidatePath || !parentPath) {
    return false;
  }
  const relative = pathApi.relative(parentPath.key, candidatePath.key);
  return relative === "" || relative !== ".." && !relative.startsWith(`..${pathApi.sep}`) && !pathApi.isAbsolute(relative);
}
async function mapLimit(items, limit, fn) {
  let cursor = 0;
  async function worker() {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      await fn(items[index]);
    }
  }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, () => worker()));
}
async function scanGitRepos(roots, options = {}) {
  if (options.enabled === false) {
    return [];
  }
  const maxDepthValue = Number(options.maxDepth);
  const maxDepth = Number.isFinite(maxDepthValue) && maxDepthValue >= 0 ? maxDepthValue : DEFAULT_MAX_DEPTH;
  const pathOptions = {};
  const requestedRoots = Array.isArray(roots) && roots.length > 0 ? roots : [os2.homedir()];
  const searchRoots = [
    ...new Map(
      requestedRoots.map((root) => normalizeRepoScanPath(root, pathOptions)).filter((entry) => entry !== null).map((entry) => [entry.key, entry.value])
    ).values()
  ];
  const exclusions = (options.excludePaths ?? []).map((excluded) => normalizeRepoScanPath(excluded, pathOptions)).filter((entry) => entry !== null);
  const found = /* @__PURE__ */ new Map();
  function isExcluded(candidate) {
    return exclusions.some((excluded) => repoScanPathIsWithin(candidate, excluded.value, pathOptions));
  }
  async function walk(dir, depth) {
    if (depth > maxDepth || isExcluded(dir)) {
      return;
    }
    let entries;
    try {
      entries = await fsp2.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    const gitDir = entries.find((entry) => entry.name === ".git" && entry.isDirectory());
    if (gitDir) {
      try {
        await fsp2.access(path8.join(dir, ".git", "HEAD"), fs8.constants.R_OK);
      } catch {
        return;
      }
      const normalized = normalizeRepoScanPath(dir, pathOptions);
      if (normalized) {
        found.set(normalized.key, {
          root: normalized.value,
          label: path8.basename(normalized.value) || normalized.value
        });
      }
      return;
    }
    const subdirs = entries.filter((entry) => entry.isDirectory() && !entry.name.startsWith(".") && !JUNK_DIRS.has(entry.name)).map((entry) => path8.join(dir, entry.name));
    await mapLimit(subdirs, MAX_CONCURRENCY, (subdir) => walk(subdir, depth + 1));
  }
  await mapLimit(searchRoots, MAX_CONCURRENCY, (root) => walk(root, 0));
  return [...found.values()];
}

// electron/git-review-ops.ts
import { execFile } from "node:child_process";
import fs9 from "node:fs/promises";
import path9 from "node:path";

// ../../node_modules/simple-git/dist/esm/index.js
var import_file_exists = __toESM(require_dist(), 1);

// ../../node_modules/@simple-git/args-pathspec/dist/index.mjs
var t = /* @__PURE__ */ new WeakMap();
function c(...n) {
  const e = new String(n);
  return t.set(e, n), e;
}
function r(n) {
  return n instanceof String && t.has(n);
}
function o(n) {
  return t.get(n) ?? [];
}

// ../../node_modules/simple-git/dist/esm/index.js
var import_debug = __toESM(require_src(), 1);
import { spawn as spawn2 } from "child_process";
var import_promise_deferred = __toESM(require_dist2(), 1);
import { normalize } from "node:path";

// ../../node_modules/@simple-git/argv-parser/dist/index.mjs
function* U(e, t2) {
  const n = t2 === "global";
  for (const o2 of e)
    o2.isGlobal === n && (yield o2);
}
var k = /* @__PURE__ */ new Set([
  "--add",
  "--edit",
  "--remove-section",
  "--rename-section",
  "--replace-all",
  "--unset",
  "--unset-all",
  "-e"
]);
var S = /* @__PURE__ */ new Set([
  "--get",
  "--get-all",
  "--get-color",
  "--get-colorbool",
  "--get-regexp",
  "--get-urlmatch",
  "--list",
  "-l"
]);
var P = /* @__PURE__ */ new Set([
  "edit",
  "remove-section",
  "rename-section",
  "set",
  "unset"
]);
var E = /* @__PURE__ */ new Set(["get", "get-color", "get-colorbool", "list"]);
function F(e, t2) {
  for (const { name: o2 } of U(e, "task")) {
    if (k.has(o2))
      return p(true, t2);
    if (S.has(o2))
      return p(false, t2);
  }
  const n = t2.at(0)?.toLowerCase();
  return n === void 0 ? null : P.has(n) ? p(true, t2.slice(1)) : E.has(n) ? p(false, t2.slice(1)) : t2.length === 1 ? p(false, t2) : p(true, t2);
}
function p(e = false, t2 = []) {
  const n = t2.at(0)?.toLowerCase();
  return n === void 0 ? null : {
    isWrite: e,
    isRead: !e,
    key: n,
    value: t2.at(1)
  };
}
function A(e, t2) {
  return t2.isWrite && t2.value !== void 0 ? { key: t2.key, value: t2.value, scope: e } : { key: t2.key, scope: e };
}
function M(e) {
  const t2 = e?.indexOf("=") || -1;
  return !e || t2 < 0 ? null : {
    key: e.slice(0, t2).trim().toLowerCase(),
    value: e.slice(t2 + 1)
  };
}
function N(e) {
  for (const { name: t2 } of U(e, "task"))
    switch (t2) {
      case "--global":
        return "global";
      case "--system":
        return "system";
      case "--worktree":
        return "worktree";
      case "--local":
        return "local";
      case "--file":
      case "-f":
        return "file";
    }
  return "local";
}
function G({ name: e }) {
  if (e === "-c" || e === "--config")
    return "inline";
  if (e === "--config-env")
    return "env";
}
function* O(e) {
  for (const t2 of e) {
    const n = G(t2), o2 = n && M(t2.value);
    o2 && (yield {
      ...o2,
      scope: n
    });
  }
}
function L(e, t2, n) {
  const o2 = {
    read: [],
    write: [...O(t2)]
  };
  return e === "config" && $(
    o2,
    N(t2),
    F(t2, n)
  ), o2;
}
function $(e, t2, n) {
  if (n === null)
    return;
  const o2 = A(t2, n);
  n.isWrite ? e.write.push(o2) : e.read.push(o2);
}
var x = {
  short: /* @__PURE__ */ new Map([
    ["c", true]
    //  -c <k=v>    set config key for this invocation
  ])
};
var D = {
  short: new Map([
    ["C", true],
    //  -C <path>   change working directory
    ["P", false],
    // -P          no pager (alias for --no-pager)
    ["h", false],
    // -h          help
    ["p", false],
    // -p          paginate
    ["v", false],
    // -v          version
    ...x.short.entries()
  ]),
  long: /* @__PURE__ */ new Set([
    "attr-source",
    "config-env",
    "exec-path",
    "git-dir",
    "list-cmds",
    "namespace",
    "super-prefix",
    "work-tree"
  ])
};
var R = {
  clone: {
    short: /* @__PURE__ */ new Map([
      ["b", true],
      // -b <branch>
      ["j", true],
      // -j <n>          parallel jobs
      ["l", false],
      // -l local
      ["n", false],
      // -n no-checkout
      ["o", true],
      // -o <name>       remote name
      ["q", false],
      // -q quiet
      ["s", false],
      // -s shared
      ["u", true]
      // -u <upload-pack>
    ]),
    long: /* @__PURE__ */ new Set(["branch", "config", "jobs", "origin", "upload-pack", "u", "template"])
  },
  commit: {
    short: /* @__PURE__ */ new Map([
      ["C", true],
      // -C <commit>  reuse message
      ["F", true],
      // -F <file>    read message from file
      ["c", true],
      // -c <commit>  reedit message
      ["m", true],
      // -m <msg>
      ["t", true]
      // -t <template>
    ]),
    long: /* @__PURE__ */ new Set(["file", "message", "reedit-message", "reuse-message", "template"])
  },
  config: {
    short: /* @__PURE__ */ new Map([
      ["e", false],
      // -e  open editor
      ["f", true],
      //  -f <file>
      ["l", false]
      // -l  list
    ]),
    long: /* @__PURE__ */ new Set(["blob", "comment", "default", "file", "type", "value"])
  },
  fetch: {
    short: /* @__PURE__ */ new Map(),
    long: /* @__PURE__ */ new Set(["upload-pack"])
  },
  init: {
    short: /* @__PURE__ */ new Map(),
    long: /* @__PURE__ */ new Set(["template"])
  },
  pull: {
    short: /* @__PURE__ */ new Map(),
    long: /* @__PURE__ */ new Set(["upload-pack"])
  },
  push: {
    short: /* @__PURE__ */ new Map(),
    long: /* @__PURE__ */ new Set(["exec", "receive-pack"])
  }
};
var T = { short: /* @__PURE__ */ new Map(), long: /* @__PURE__ */ new Set() };
function I(e) {
  const t2 = R[e ?? ""] ?? T;
  return {
    short: new Map([...x.short.entries(), ...t2.short.entries()]),
    long: t2.long
  };
}
function b(e, t2 = D) {
  if (e.startsWith("--")) {
    const n = e.indexOf("=");
    if (n > 2)
      return [{ name: e.slice(0, n), value: e.slice(n + 1), needsNext: false }];
    const o2 = e.slice(2);
    return [{ name: e, needsNext: t2.long.has(o2) }];
  }
  if (e.length === 2) {
    const n = e.charAt(1), o2 = t2.short.get(n);
    return [{ name: e, needsNext: o2 === true }];
  }
  return W(e, t2.short);
}
function W(e, t2) {
  const n = e.slice(1).split(""), o2 = [];
  for (let s = 0; s < n.length; s++) {
    const r2 = n[s], l = t2.get(r2);
    if (l === void 0)
      return [{ name: e, needsNext: false }];
    if (l) {
      const a = n.slice(s + 1).join("");
      if (a && ![...a].every((w) => t2.has(w)))
        return o2.push({ name: `-${r2}`, value: a, needsNext: false }), o2;
    }
    o2.push({ name: `-${r2}`, needsNext: l });
  }
  return o2;
}
function j(e, t2 = []) {
  let n = 0;
  for (; n < e.length; ) {
    const o2 = String(e[n]);
    if (!o2.startsWith("-") || o2.length < 2) break;
    const s = b(o2);
    let r2 = n + 1;
    for (const l of s) {
      const a = {
        name: l.name,
        value: l.value,
        absorbedNext: false,
        isGlobal: true
      };
      l.needsNext && a.value === void 0 && r2 < e.length && (a.value = String(e[r2]), a.absorbedNext = true, r2++), t2.push(a);
    }
    n = r2;
  }
  return { flags: t2, taskIndex: n };
}
function B(e, t2, n = []) {
  const o2 = I(t2), s = [], r2 = [];
  let l = 0;
  for (; l < e.length; ) {
    const a = e[l];
    if (r(a)) {
      r2.push(...o(a)), l++;
      continue;
    }
    const f = String(a);
    if (f === "--") {
      for (let g = l + 1; g < e.length; g++) {
        const u = e[g];
        r(u) ? r2.push(...o(u)) : r2.push(String(u));
      }
      break;
    }
    if (!f.startsWith("-") || f.length < 2) {
      s.push(f), l++;
      continue;
    }
    const w = b(f, o2);
    let d = l + 1;
    for (const g of w) {
      const u = {
        name: g.name,
        value: g.value,
        absorbedNext: false,
        isGlobal: false
      };
      g.needsNext && u.value === void 0 && d < e.length && !r(e[d]) && (u.value = String(e[d]), u.absorbedNext = true, d++), n.push(u);
    }
    l = d;
  }
  return { flags: n, positionals: s, pathspecs: r2 };
}
function* V({
  write: e
}) {
  for (const t2 of e)
    for (const n of q) {
      const o2 = n(t2.key);
      o2 && (yield o2);
    }
}
function c2(e, t2, n = String(e)) {
  const o2 = typeof e == "string" ? new RegExp(`\\s*${e.toLowerCase()}`) : e;
  return function(r2) {
    if (o2.test(r2))
      return {
        category: t2,
        message: `Configuring ${n} is not permitted without enabling ${t2}`
      };
  };
}
function i(e, t2) {
  const n = new RegExp(`\\s*${e.toLowerCase().replace(/\./g, "(..+)?.")}`);
  return c2(n, t2, e);
}
var q = [
  c2("alias", "allowUnsafeAlias"),
  c2("core.askPass", "allowUnsafeAskPass"),
  c2("core.editor", "allowUnsafeEditor"),
  c2("core.fsmonitor", "allowUnsafeFsMonitor"),
  c2("core.gitProxy", "allowUnsafeGitProxy"),
  c2("core.hooksPath", "allowUnsafeHooksPath"),
  c2("core.pager", "allowUnsafePager"),
  c2("core.sshCommand", "allowUnsafeSshCommand"),
  i("credential.helper", "allowUnsafeCredentialHelper"),
  i("diff.command", "allowUnsafeDiffExternal"),
  c2("diff.external", "allowUnsafeDiffExternal"),
  i("diff.textconv", "allowUnsafeDiffTextConv"),
  i("filter.clean", "allowUnsafeFilter"),
  i("filter.smudge", "allowUnsafeFilter"),
  i("gpg.program", "allowUnsafeGpgProgram"),
  c2("init.templateDir", "allowUnsafeTemplateDir"),
  i("merge.driver", "allowUnsafeMergeDriver"),
  i("mergetool.path", "allowUnsafeMergeDriver"),
  i("mergetool.cmd", "allowUnsafeMergeDriver"),
  i("protocol.allow", "allowUnsafeProtocolOverride"),
  i("remote.receivepack", "allowUnsafePack"),
  i("remote.uploadpack", "allowUnsafePack"),
  c2("sequence.editor", "allowUnsafeEditor")
];
function* K(e, t2) {
  for (const n of t2)
    for (const o2 of H) {
      const s = o2(e, n.name);
      s && (yield s);
    }
}
function h(e, t2, n, o2 = String(t2)) {
  const s = typeof t2 == "string" ? new RegExp(`\\s*${t2.toLowerCase()}`) : t2, r2 = `Use of ${e ? `${e} with option ` : ""}${o2} is not permitted without enabling ${n}`;
  return function(a, f) {
    if ((!e || a === e) && s.test(f))
      return {
        category: n,
        message: r2
      };
  };
}
var H = [
  h(
    null,
    /--(upload|receive)-pack/,
    "allowUnsafePack",
    "--upload-pack or --receive-pack"
  ),
  h("clone", /^-\w*u/, "allowUnsafePack"),
  h("clone", "--u", "allowUnsafePack"),
  h("push", "--exec", "allowUnsafePack"),
  h(null, "--template", "allowUnsafeTemplateDir")
];
function C(e, t2, n) {
  return [...K(e, t2), ...V(n)];
}
function Y(...e) {
  const { flags: t2, taskIndex: n } = j(e), o2 = n < e.length ? String(e[n]).toLowerCase() : null, s = o2 !== null ? e.slice(n + 1) : [], { positionals: r2, pathspecs: l } = B(s, o2, t2), a = L(o2, t2, r2);
  return {
    task: o2,
    flags: t2.map(J),
    paths: l,
    config: a,
    vulnerabilities: z(C(o2, t2, a))
  };
}
function z(e) {
  return Object.defineProperty(e, "vulnerabilities", {
    value: e
  });
}
function J({ value: e, name: t2 }) {
  return e !== void 0 ? { name: t2, value: e } : { name: t2 };
}
var y = {
  editor: "allowUnsafeEditor",
  git_askpass: "allowUnsafeAskPass",
  git_config_global: "allowUnsafeConfigPaths",
  git_config_system: "allowUnsafeConfigPaths",
  git_config_count: "allowUnsafeConfigEnvCount",
  git_config: "allowUnsafeConfigPaths",
  git_editor: "allowUnsafeEditor",
  git_exec_path: "allowUnsafeConfigPaths",
  git_external_diff: "allowUnsafeDiffExternal",
  git_pager: "allowUnsafePager",
  git_proxy_command: "allowUnsafeGitProxy",
  git_template_dir: "allowUnsafeTemplateDir",
  git_sequence_editor: "allowUnsafeEditor",
  git_ssh: "allowUnsafeSshCommand",
  git_ssh_command: "allowUnsafeSshCommand",
  pager: "allowUnsafePager",
  prefix: "allowUnsafeConfigPaths",
  ssh_askpass: "allowUnsafeAskPass"
};
function* Q(e) {
  const t2 = parseInt(e.git_config_count ?? "0", 10);
  for (let n = 0; n < t2; n++) {
    const o2 = e[`git_config_key_${n}`], s = e[`git_config_value_${n}`];
    o2 !== void 0 && (yield { key: o2.toLowerCase().trim(), value: s, scope: "env" });
  }
}
function* X(e) {
  for (const t2 of Object.keys(e))
    if (_(t2)) {
      const n = y[t2];
      yield {
        category: n,
        message: `Use of "${t2.toUpperCase()}" is not permitted without enabling ${n}`
      };
    }
}
function _(e) {
  return Object.hasOwn(y, e);
}
function Z(e) {
  const t2 = {};
  for (const [n, o2] of Object.entries(e)) {
    const s = n.toLowerCase().trim();
    (_(s) || s.startsWith("git")) && (t2[s] = String(o2));
  }
  return t2;
}
function ee(e) {
  const t2 = Z(e), n = {
    read: [],
    write: [...Q(t2)]
  }, o2 = [
    ...X(t2),
    ...C(null, [], n)
  ];
  return {
    config: n,
    vulnerabilities: o2
  };
}
function ne(e, t2) {
  return [...Y(...e).vulnerabilities, ...ee(t2).vulnerabilities];
}

// ../../node_modules/simple-git/dist/esm/index.js
var import_promise_deferred2 = __toESM(require_dist2(), 1);
import { EventEmitter } from "node:events";
var __defProp2 = Object.defineProperty;
var __getOwnPropDesc2 = Object.getOwnPropertyDescriptor;
var __getOwnPropNames2 = Object.getOwnPropertyNames;
var __hasOwnProp2 = Object.prototype.hasOwnProperty;
var __esm2 = (fn, res) => function __init() {
  return fn && (res = (0, fn[__getOwnPropNames2(fn)[0]])(fn = 0)), res;
};
var __commonJS2 = (cb, mod) => function __require2() {
  return mod || (0, cb[__getOwnPropNames2(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
};
var __export2 = (target2, all) => {
  for (var name in all)
    __defProp2(target2, name, { get: all[name], enumerable: true });
};
var __copyProps2 = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames2(from))
      if (!__hasOwnProp2.call(to, key) && key !== except)
        __defProp2(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc2(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS2 = (mod) => __copyProps2(__defProp2({}, "__esModule", { value: true }), mod);
var GitError;
var init_git_error = __esm2({
  "src/lib/errors/git-error.ts"() {
    "use strict";
    GitError = class extends Error {
      constructor(task, message) {
        super(message);
        this.task = task;
        Object.setPrototypeOf(this, new.target.prototype);
      }
    };
  }
});
var GitResponseError;
var init_git_response_error = __esm2({
  "src/lib/errors/git-response-error.ts"() {
    "use strict";
    init_git_error();
    GitResponseError = class extends GitError {
      constructor(git, message) {
        super(void 0, message || String(git));
        this.git = git;
      }
    };
  }
});
var TaskConfigurationError;
var init_task_configuration_error = __esm2({
  "src/lib/errors/task-configuration-error.ts"() {
    "use strict";
    init_git_error();
    TaskConfigurationError = class extends GitError {
      constructor(message) {
        super(void 0, message);
      }
    };
  }
});
function asFunction(source) {
  if (typeof source !== "function") {
    return NOOP;
  }
  return source;
}
function isUserFunction(source) {
  return typeof source === "function" && source !== NOOP;
}
function splitOn(input, char) {
  const index = input.indexOf(char);
  if (index <= 0) {
    return [input, ""];
  }
  return [input.substr(0, index), input.substr(index + 1)];
}
function first(input, offset = 0) {
  return isArrayLike(input) && input.length > offset ? input[offset] : void 0;
}
function last(input, offset = 0) {
  if (isArrayLike(input) && input.length > offset) {
    return input[input.length - 1 - offset];
  }
}
function isArrayLike(input) {
  return filterHasLength(input);
}
function toLinesWithContent(input = "", trimmed2 = true, separator = "\n") {
  return input.split(separator).reduce((output, line) => {
    const lineContent = trimmed2 ? line.trim() : line;
    if (lineContent) {
      output.push(lineContent);
    }
    return output;
  }, []);
}
function forEachLineWithContent(input, callback) {
  return toLinesWithContent(input, true).map((line) => callback(line));
}
function folderExists(path23) {
  return (0, import_file_exists.exists)(path23, import_file_exists.FOLDER);
}
function append(target2, item) {
  if (Array.isArray(target2)) {
    if (!target2.includes(item)) {
      target2.push(item);
    }
  } else {
    target2.add(item);
  }
  return item;
}
function including(target2, item) {
  if (Array.isArray(target2) && !target2.includes(item)) {
    target2.push(item);
  }
  return target2;
}
function remove(target2, item) {
  if (Array.isArray(target2)) {
    const index = target2.indexOf(item);
    if (index >= 0) {
      target2.splice(index, 1);
    }
  } else {
    target2.delete(item);
  }
  return item;
}
function asArray(source) {
  return Array.isArray(source) ? source : [source];
}
function asCamelCase(str) {
  return str.replace(/[\s-]+(.)/g, (_all, chr) => {
    return chr.toUpperCase();
  });
}
function asStringArray(source) {
  return asArray(source).map((item) => {
    return item instanceof String ? item : String(item);
  });
}
function asNumber(source, onNaN = 0) {
  if (source == null) {
    return onNaN;
  }
  const num = parseInt(source, 10);
  return Number.isNaN(num) ? onNaN : num;
}
function prefixedArray(input, prefix) {
  const output = [];
  for (let i2 = 0, max = input.length; i2 < max; i2++) {
    output.push(prefix, input[i2]);
  }
  return output;
}
function bufferToString(input) {
  return (Array.isArray(input) ? Buffer.concat(input) : input).toString("utf-8");
}
function pick(source, properties) {
  const out = {};
  properties.forEach((key) => {
    if (source[key] !== void 0) {
      out[key] = source[key];
    }
  });
  return out;
}
function delay(duration = 0) {
  return new Promise((done) => setTimeout(done, duration));
}
function orVoid(input) {
  if (input === false) {
    return void 0;
  }
  return input;
}
var NULL;
var NOOP;
var objectToString;
var init_util = __esm2({
  "src/lib/utils/util.ts"() {
    "use strict";
    init_argument_filters();
    NULL = "\0";
    NOOP = () => {
    };
    objectToString = Object.prototype.toString.call.bind(Object.prototype.toString);
  }
});
function filterType(input, filter, def) {
  if (filter(input)) {
    return input;
  }
  return arguments.length > 2 ? def : void 0;
}
function filterPrimitives(input, omit) {
  const type = r(input) ? "string" : typeof input;
  return /number|string|boolean/.test(type) && (!omit || !omit.includes(type));
}
function filterPlainObject(input) {
  return !!input && objectToString(input) === "[object Object]";
}
function filterFunction(input) {
  return typeof input === "function";
}
var filterArray;
var filterNumber;
var filterString;
var filterStringOrStringArray;
var filterHasLength;
var init_argument_filters = __esm2({
  "src/lib/utils/argument-filters.ts"() {
    "use strict";
    init_util();
    filterArray = (input) => {
      return Array.isArray(input);
    };
    filterNumber = (input) => {
      return typeof input === "number";
    };
    filterString = (input) => {
      return typeof input === "string" || r(input);
    };
    filterStringOrStringArray = (input) => {
      return filterString(input) || Array.isArray(input) && input.every(filterString);
    };
    filterHasLength = (input) => {
      if (input == null || "number|boolean|function".includes(typeof input)) {
        return false;
      }
      return typeof input.length === "number";
    };
  }
});
var ExitCodes;
var init_exit_codes = __esm2({
  "src/lib/utils/exit-codes.ts"() {
    "use strict";
    ExitCodes = /* @__PURE__ */ ((ExitCodes2) => {
      ExitCodes2[ExitCodes2["SUCCESS"] = 0] = "SUCCESS";
      ExitCodes2[ExitCodes2["ERROR"] = 1] = "ERROR";
      ExitCodes2[ExitCodes2["NOT_FOUND"] = -2] = "NOT_FOUND";
      ExitCodes2[ExitCodes2["UNCLEAN"] = 128] = "UNCLEAN";
      return ExitCodes2;
    })(ExitCodes || {});
  }
});
var GitOutputStreams;
var init_git_output_streams = __esm2({
  "src/lib/utils/git-output-streams.ts"() {
    "use strict";
    GitOutputStreams = class _GitOutputStreams {
      constructor(stdOut, stdErr) {
        this.stdOut = stdOut;
        this.stdErr = stdErr;
      }
      asStrings() {
        return new _GitOutputStreams(this.stdOut.toString("utf8"), this.stdErr.toString("utf8"));
      }
    };
  }
});
function useMatchesDefault() {
  throw new Error(`LineParser:useMatches not implemented`);
}
var LineParser;
var RemoteLineParser;
var init_line_parser = __esm2({
  "src/lib/utils/line-parser.ts"() {
    "use strict";
    LineParser = class {
      constructor(regExp, useMatches) {
        this.matches = [];
        this.useMatches = useMatchesDefault;
        this.parse = (line, target2) => {
          this.resetMatches();
          if (!this._regExp.every((reg, index) => this.addMatch(reg, index, line(index)))) {
            return false;
          }
          return this.useMatches(target2, this.prepareMatches()) !== false;
        };
        this._regExp = Array.isArray(regExp) ? regExp : [regExp];
        if (useMatches) {
          this.useMatches = useMatches;
        }
      }
      resetMatches() {
        this.matches.length = 0;
      }
      prepareMatches() {
        return this.matches;
      }
      addMatch(reg, index, line) {
        const matched = line && reg.exec(line);
        if (matched) {
          this.pushMatch(index, matched);
        }
        return !!matched;
      }
      pushMatch(_index, matched) {
        this.matches.push(...matched.slice(1));
      }
    };
    RemoteLineParser = class extends LineParser {
      addMatch(reg, index, line) {
        return /^remote:\s/.test(String(line)) && super.addMatch(reg, index, line);
      }
      pushMatch(index, matched) {
        if (index > 0 || matched.length > 1) {
          super.pushMatch(index, matched);
        }
      }
    };
  }
});
function createInstanceConfig(...options) {
  const baseDir = process.cwd();
  const config = Object.assign(
    { baseDir, ...defaultOptions },
    ...options.filter((o2) => typeof o2 === "object" && o2)
  );
  config.baseDir = config.baseDir || baseDir;
  config.trimmed = config.trimmed === true;
  return config;
}
var defaultOptions;
var init_simple_git_options = __esm2({
  "src/lib/utils/simple-git-options.ts"() {
    "use strict";
    defaultOptions = {
      binary: "git",
      maxConcurrentProcesses: 5,
      config: [],
      trimmed: false
    };
  }
});
function appendTaskOptions(options, commands = []) {
  if (!filterPlainObject(options)) {
    return commands;
  }
  return Object.keys(options).reduce((commands2, key) => {
    const value = options[key];
    if (r(value)) {
      commands2.push(value);
    } else if (filterPrimitives(value, ["boolean"])) {
      commands2.push(key + "=" + value);
    } else if (Array.isArray(value)) {
      for (const v of value) {
        if (!filterPrimitives(v, ["string", "number"])) {
          commands2.push(key + "=" + v);
        }
      }
    } else {
      commands2.push(key);
    }
    return commands2;
  }, commands);
}
function getTrailingOptions(args, initialPrimitive = 0, objectOnly = false) {
  const command = [];
  for (let i2 = 0, max = initialPrimitive < 0 ? args.length : initialPrimitive; i2 < max; i2++) {
    if ("string|number".includes(typeof args[i2])) {
      command.push(String(args[i2]));
    }
  }
  appendTaskOptions(trailingOptionsArgument(args), command);
  if (!objectOnly) {
    command.push(...trailingArrayArgument(args));
  }
  return command;
}
function trailingArrayArgument(args) {
  const hasTrailingCallback = typeof last(args) === "function";
  return asStringArray(filterType(last(args, hasTrailingCallback ? 1 : 0), filterArray, []));
}
function trailingOptionsArgument(args) {
  const hasTrailingCallback = filterFunction(last(args));
  return filterType(last(args, hasTrailingCallback ? 1 : 0), filterPlainObject);
}
function trailingFunctionArgument(args, includeNoop = true) {
  const callback = asFunction(last(args));
  return includeNoop || isUserFunction(callback) ? callback : void 0;
}
var init_task_options = __esm2({
  "src/lib/utils/task-options.ts"() {
    "use strict";
    init_argument_filters();
    init_util();
  }
});
function callTaskParser(parser4, streams) {
  return parser4(streams.stdOut, streams.stdErr);
}
function parseStringResponse(result, parsers12, texts, trim = true) {
  asArray(texts).forEach((text) => {
    for (let lines = toLinesWithContent(text, trim), i2 = 0, max = lines.length; i2 < max; i2++) {
      const line = (offset = 0) => {
        if (i2 + offset >= max) {
          return;
        }
        return lines[i2 + offset];
      };
      parsers12.some(({ parse }) => parse(line, result));
    }
  });
  return result;
}
var init_task_parser = __esm2({
  "src/lib/utils/task-parser.ts"() {
    "use strict";
    init_util();
  }
});
var utils_exports = {};
__export2(utils_exports, {
  ExitCodes: () => ExitCodes,
  GitOutputStreams: () => GitOutputStreams,
  LineParser: () => LineParser,
  NOOP: () => NOOP,
  NULL: () => NULL,
  RemoteLineParser: () => RemoteLineParser,
  append: () => append,
  appendTaskOptions: () => appendTaskOptions,
  asArray: () => asArray,
  asCamelCase: () => asCamelCase,
  asFunction: () => asFunction,
  asNumber: () => asNumber,
  asStringArray: () => asStringArray,
  bufferToString: () => bufferToString,
  callTaskParser: () => callTaskParser,
  createInstanceConfig: () => createInstanceConfig,
  delay: () => delay,
  filterArray: () => filterArray,
  filterFunction: () => filterFunction,
  filterHasLength: () => filterHasLength,
  filterNumber: () => filterNumber,
  filterPlainObject: () => filterPlainObject,
  filterPrimitives: () => filterPrimitives,
  filterString: () => filterString,
  filterStringOrStringArray: () => filterStringOrStringArray,
  filterType: () => filterType,
  first: () => first,
  folderExists: () => folderExists,
  forEachLineWithContent: () => forEachLineWithContent,
  getTrailingOptions: () => getTrailingOptions,
  including: () => including,
  isUserFunction: () => isUserFunction,
  last: () => last,
  objectToString: () => objectToString,
  orVoid: () => orVoid,
  parseStringResponse: () => parseStringResponse,
  pick: () => pick,
  prefixedArray: () => prefixedArray,
  remove: () => remove,
  splitOn: () => splitOn,
  toLinesWithContent: () => toLinesWithContent,
  trailingFunctionArgument: () => trailingFunctionArgument,
  trailingOptionsArgument: () => trailingOptionsArgument
});
var init_utils = __esm2({
  "src/lib/utils/index.ts"() {
    "use strict";
    init_argument_filters();
    init_exit_codes();
    init_git_output_streams();
    init_line_parser();
    init_simple_git_options();
    init_task_options();
    init_task_parser();
    init_util();
  }
});
var check_is_repo_exports = {};
__export2(check_is_repo_exports, {
  CheckRepoActions: () => CheckRepoActions,
  checkIsBareRepoTask: () => checkIsBareRepoTask,
  checkIsRepoRootTask: () => checkIsRepoRootTask,
  checkIsRepoTask: () => checkIsRepoTask
});
function checkIsRepoTask(action) {
  switch (action) {
    case "bare":
      return checkIsBareRepoTask();
    case "root":
      return checkIsRepoRootTask();
  }
  const commands = ["rev-parse", "--is-inside-work-tree"];
  return {
    commands,
    format: "utf-8",
    onError,
    parser
  };
}
function checkIsRepoRootTask() {
  const commands = ["rev-parse", "--git-dir"];
  return {
    commands,
    format: "utf-8",
    onError,
    parser(path23) {
      return /^\.(git)?$/.test(path23.trim());
    }
  };
}
function checkIsBareRepoTask() {
  const commands = ["rev-parse", "--is-bare-repository"];
  return {
    commands,
    format: "utf-8",
    onError,
    parser
  };
}
function isNotRepoMessage(error) {
  return /(Not a git repository|Kein Git-Repository)/i.test(String(error));
}
var CheckRepoActions;
var onError;
var parser;
var init_check_is_repo = __esm2({
  "src/lib/tasks/check-is-repo.ts"() {
    "use strict";
    init_utils();
    CheckRepoActions = /* @__PURE__ */ ((CheckRepoActions2) => {
      CheckRepoActions2["BARE"] = "bare";
      CheckRepoActions2["IN_TREE"] = "tree";
      CheckRepoActions2["IS_REPO_ROOT"] = "root";
      return CheckRepoActions2;
    })(CheckRepoActions || {});
    onError = ({ exitCode }, error, done, fail) => {
      if (exitCode === 128 && isNotRepoMessage(error)) {
        return done(Buffer.from("false"));
      }
      fail(error);
    };
    parser = (text) => {
      return text.trim() === "true";
    };
  }
});
function cleanSummaryParser(dryRun, text) {
  const summary = new CleanResponse(dryRun);
  const regexp = dryRun ? dryRunRemovalRegexp : removalRegexp;
  toLinesWithContent(text).forEach((line) => {
    const removed = line.replace(regexp, "");
    summary.paths.push(removed);
    (isFolderRegexp.test(removed) ? summary.folders : summary.files).push(removed);
  });
  return summary;
}
var CleanResponse;
var removalRegexp;
var dryRunRemovalRegexp;
var isFolderRegexp;
var init_CleanSummary = __esm2({
  "src/lib/responses/CleanSummary.ts"() {
    "use strict";
    init_utils();
    CleanResponse = class {
      constructor(dryRun) {
        this.dryRun = dryRun;
        this.paths = [];
        this.files = [];
        this.folders = [];
      }
    };
    removalRegexp = /^[a-z]+\s*/i;
    dryRunRemovalRegexp = /^[a-z]+\s+[a-z]+\s*/i;
    isFolderRegexp = /\/$/;
  }
});
var task_exports = {};
__export2(task_exports, {
  EMPTY_COMMANDS: () => EMPTY_COMMANDS,
  adhocExecTask: () => adhocExecTask,
  configurationErrorTask: () => configurationErrorTask,
  isBufferTask: () => isBufferTask,
  isEmptyTask: () => isEmptyTask,
  straightThroughBufferTask: () => straightThroughBufferTask,
  straightThroughStringTask: () => straightThroughStringTask
});
function adhocExecTask(parser4) {
  return {
    commands: EMPTY_COMMANDS,
    format: "empty",
    parser: parser4
  };
}
function configurationErrorTask(error) {
  return {
    commands: EMPTY_COMMANDS,
    format: "empty",
    parser() {
      throw typeof error === "string" ? new TaskConfigurationError(error) : error;
    }
  };
}
function straightThroughStringTask(commands, trimmed2 = false) {
  return {
    commands,
    format: "utf-8",
    parser(text) {
      return trimmed2 ? String(text).trim() : text;
    }
  };
}
function straightThroughBufferTask(commands) {
  return {
    commands,
    format: "buffer",
    parser(buffer) {
      return buffer;
    }
  };
}
function isBufferTask(task) {
  return task.format === "buffer";
}
function isEmptyTask(task) {
  return task.format === "empty" || !task.commands.length;
}
var EMPTY_COMMANDS;
var init_task = __esm2({
  "src/lib/tasks/task.ts"() {
    "use strict";
    init_task_configuration_error();
    EMPTY_COMMANDS = [];
  }
});
var clean_exports = {};
__export2(clean_exports, {
  CONFIG_ERROR_INTERACTIVE_MODE: () => CONFIG_ERROR_INTERACTIVE_MODE,
  CONFIG_ERROR_MODE_REQUIRED: () => CONFIG_ERROR_MODE_REQUIRED,
  CONFIG_ERROR_UNKNOWN_OPTION: () => CONFIG_ERROR_UNKNOWN_OPTION,
  CleanOptions: () => CleanOptions,
  cleanTask: () => cleanTask,
  cleanWithOptionsTask: () => cleanWithOptionsTask,
  isCleanOptionsArray: () => isCleanOptionsArray
});
function cleanWithOptionsTask(mode, customArgs) {
  const { cleanMode, options, valid } = getCleanOptions(mode);
  if (!cleanMode) {
    return configurationErrorTask(CONFIG_ERROR_MODE_REQUIRED);
  }
  if (!valid.options) {
    return configurationErrorTask(CONFIG_ERROR_UNKNOWN_OPTION + JSON.stringify(mode));
  }
  options.push(...customArgs);
  if (options.some(isInteractiveMode)) {
    return configurationErrorTask(CONFIG_ERROR_INTERACTIVE_MODE);
  }
  return cleanTask(cleanMode, options);
}
function cleanTask(mode, customArgs) {
  const commands = ["clean", `-${mode}`, ...customArgs];
  return {
    commands,
    format: "utf-8",
    parser(text) {
      return cleanSummaryParser(mode === "n", text);
    }
  };
}
function isCleanOptionsArray(input) {
  return Array.isArray(input) && input.every((test) => CleanOptionValues.has(test));
}
function getCleanOptions(input) {
  let cleanMode;
  let options = [];
  let valid = { cleanMode: false, options: true };
  input.replace(/[^a-z]i/g, "").split("").forEach((char) => {
    if (isCleanMode(char)) {
      cleanMode = char;
      valid.cleanMode = true;
    } else {
      valid.options = valid.options && isKnownOption(options[options.length] = `-${char}`);
    }
  });
  return {
    cleanMode,
    options,
    valid
  };
}
function isCleanMode(cleanMode) {
  return cleanMode === "f" || cleanMode === "n";
}
function isKnownOption(option) {
  return /^-[a-z]$/i.test(option) && CleanOptionValues.has(option.charAt(1));
}
function isInteractiveMode(option) {
  if (/^-[^\-]/.test(option)) {
    return option.indexOf("i") > 0;
  }
  return option === "--interactive";
}
var CONFIG_ERROR_INTERACTIVE_MODE;
var CONFIG_ERROR_MODE_REQUIRED;
var CONFIG_ERROR_UNKNOWN_OPTION;
var CleanOptions;
var CleanOptionValues;
var init_clean = __esm2({
  "src/lib/tasks/clean.ts"() {
    "use strict";
    init_CleanSummary();
    init_utils();
    init_task();
    CONFIG_ERROR_INTERACTIVE_MODE = "Git clean interactive mode is not supported";
    CONFIG_ERROR_MODE_REQUIRED = 'Git clean mode parameter ("n" or "f") is required';
    CONFIG_ERROR_UNKNOWN_OPTION = "Git clean unknown option found in: ";
    CleanOptions = /* @__PURE__ */ ((CleanOptions2) => {
      CleanOptions2["DRY_RUN"] = "n";
      CleanOptions2["FORCE"] = "f";
      CleanOptions2["IGNORED_INCLUDED"] = "x";
      CleanOptions2["IGNORED_ONLY"] = "X";
      CleanOptions2["EXCLUDING"] = "e";
      CleanOptions2["QUIET"] = "q";
      CleanOptions2["RECURSIVE"] = "d";
      return CleanOptions2;
    })(CleanOptions || {});
    CleanOptionValues = /* @__PURE__ */ new Set([
      "i",
      ...asStringArray(Object.values(CleanOptions))
    ]);
  }
});
function configListParser(text) {
  const config = new ConfigList();
  for (const item of configParser(text)) {
    config.addValue(item.file, String(item.key), item.value);
  }
  return config;
}
function configGetParser(text, key) {
  let value = null;
  const values = [];
  const scopes = /* @__PURE__ */ new Map();
  for (const item of configParser(text, key)) {
    if (item.key !== key) {
      continue;
    }
    values.push(value = item.value);
    if (!scopes.has(item.file)) {
      scopes.set(item.file, []);
    }
    scopes.get(item.file).push(value);
  }
  return {
    key,
    paths: Array.from(scopes.keys()),
    scopes,
    value,
    values
  };
}
function configFilePath(filePath) {
  return filePath.replace(/^(file):/, "");
}
function* configParser(text, requestedKey = null) {
  const lines = text.split("\0");
  for (let i2 = 0, max = lines.length - 1; i2 < max; ) {
    const file = configFilePath(lines[i2++]);
    let value = lines[i2++];
    let key = requestedKey;
    if (value.includes("\n")) {
      const line = splitOn(value, "\n");
      key = line[0];
      value = line[1];
    }
    yield { file, key, value };
  }
}
var ConfigList;
var init_ConfigList = __esm2({
  "src/lib/responses/ConfigList.ts"() {
    "use strict";
    init_utils();
    ConfigList = class {
      constructor() {
        this.files = [];
        this.values = /* @__PURE__ */ Object.create(null);
      }
      get all() {
        if (!this._all) {
          this._all = this.files.reduce((all, file) => {
            return Object.assign(all, this.values[file]);
          }, {});
        }
        return this._all;
      }
      addFile(file) {
        if (!(file in this.values)) {
          const latest = last(this.files);
          this.values[file] = latest ? Object.create(this.values[latest]) : {};
          this.files.push(file);
        }
        return this.values[file];
      }
      addValue(file, key, value) {
        const values = this.addFile(file);
        if (!Object.hasOwn(values, key)) {
          values[key] = value;
        } else if (Array.isArray(values[key])) {
          values[key].push(value);
        } else {
          values[key] = [values[key], value];
        }
        this._all = void 0;
      }
    };
  }
});
function asConfigScope(scope, fallback) {
  if (typeof scope === "string" && Object.hasOwn(GitConfigScope, scope)) {
    return scope;
  }
  return fallback;
}
function addConfigTask(key, value, append2, scope) {
  const commands = ["config", `--${scope}`];
  if (append2) {
    commands.push("--add");
  }
  commands.push(key, value);
  return {
    commands,
    format: "utf-8",
    parser(text) {
      return text;
    }
  };
}
function getConfigTask(key, scope) {
  const commands = ["config", "--null", "--show-origin", "--get-all", key];
  if (scope) {
    commands.splice(1, 0, `--${scope}`);
  }
  return {
    commands,
    format: "utf-8",
    parser(text) {
      return configGetParser(text, key);
    }
  };
}
function listConfigTask(scope) {
  const commands = ["config", "--list", "--show-origin", "--null"];
  if (scope) {
    commands.push(`--${scope}`);
  }
  return {
    commands,
    format: "utf-8",
    parser(text) {
      return configListParser(text);
    }
  };
}
function config_default() {
  return {
    addConfig(key, value, ...rest) {
      return this._runTask(
        addConfigTask(
          key,
          value,
          rest[0] === true,
          asConfigScope(
            rest[1],
            "local"
            /* local */
          )
        ),
        trailingFunctionArgument(arguments)
      );
    },
    getConfig(key, scope) {
      return this._runTask(
        getConfigTask(key, asConfigScope(scope, void 0)),
        trailingFunctionArgument(arguments)
      );
    },
    listConfig(...rest) {
      return this._runTask(
        listConfigTask(asConfigScope(rest[0], void 0)),
        trailingFunctionArgument(arguments)
      );
    }
  };
}
var GitConfigScope;
var init_config = __esm2({
  "src/lib/tasks/config.ts"() {
    "use strict";
    init_ConfigList();
    init_utils();
    GitConfigScope = /* @__PURE__ */ ((GitConfigScope2) => {
      GitConfigScope2["system"] = "system";
      GitConfigScope2["global"] = "global";
      GitConfigScope2["local"] = "local";
      GitConfigScope2["worktree"] = "worktree";
      return GitConfigScope2;
    })(GitConfigScope || {});
  }
});
function isDiffNameStatus(input) {
  return diffNameStatus.has(input);
}
var DiffNameStatus;
var diffNameStatus;
var init_diff_name_status = __esm2({
  "src/lib/tasks/diff-name-status.ts"() {
    "use strict";
    DiffNameStatus = /* @__PURE__ */ ((DiffNameStatus2) => {
      DiffNameStatus2["ADDED"] = "A";
      DiffNameStatus2["COPIED"] = "C";
      DiffNameStatus2["DELETED"] = "D";
      DiffNameStatus2["MODIFIED"] = "M";
      DiffNameStatus2["RENAMED"] = "R";
      DiffNameStatus2["CHANGED"] = "T";
      DiffNameStatus2["UNMERGED"] = "U";
      DiffNameStatus2["UNKNOWN"] = "X";
      DiffNameStatus2["BROKEN"] = "B";
      return DiffNameStatus2;
    })(DiffNameStatus || {});
    diffNameStatus = new Set(Object.values(DiffNameStatus));
  }
});
function grepQueryBuilder(...params) {
  return new GrepQuery().param(...params);
}
function parseGrep(grep) {
  const paths = /* @__PURE__ */ new Set();
  const results = {};
  forEachLineWithContent(grep, (input) => {
    const [path23, line, preview] = input.split(NULL);
    paths.add(path23);
    (results[path23] = results[path23] || []).push({
      line: asNumber(line),
      path: path23,
      preview
    });
  });
  return {
    paths,
    results
  };
}
function grep_default() {
  return {
    grep(searchTerm) {
      const then = trailingFunctionArgument(arguments);
      const options = getTrailingOptions(arguments);
      for (const option of disallowedOptions) {
        if (options.includes(option)) {
          return this._runTask(
            configurationErrorTask(`git.grep: use of "${option}" is not supported.`),
            then
          );
        }
      }
      if (typeof searchTerm === "string") {
        searchTerm = grepQueryBuilder().param(searchTerm);
      }
      const commands = ["grep", "--null", "-n", "--full-name", ...options, ...searchTerm];
      return this._runTask(
        {
          commands,
          format: "utf-8",
          parser(stdOut) {
            return parseGrep(stdOut);
          }
        },
        then
      );
    }
  };
}
var disallowedOptions;
var Query;
var _a;
var GrepQuery;
var init_grep = __esm2({
  "src/lib/tasks/grep.ts"() {
    "use strict";
    init_utils();
    init_task();
    disallowedOptions = ["-h"];
    Query = /* @__PURE__ */ Symbol("grepQuery");
    GrepQuery = class {
      constructor() {
        this[_a] = [];
      }
      *[(_a = Query, Symbol.iterator)]() {
        for (const query of this[Query]) {
          yield query;
        }
      }
      and(...and) {
        and.length && this[Query].push("--and", "(", ...prefixedArray(and, "-e"), ")");
        return this;
      }
      param(...param) {
        this[Query].push(...prefixedArray(param, "-e"));
        return this;
      }
    };
  }
});
var reset_exports = {};
__export2(reset_exports, {
  ResetMode: () => ResetMode,
  getResetMode: () => getResetMode,
  resetTask: () => resetTask
});
function resetTask(mode, customArgs) {
  const commands = ["reset"];
  if (isValidResetMode(mode)) {
    commands.push(`--${mode}`);
  }
  commands.push(...customArgs);
  return straightThroughStringTask(commands);
}
function getResetMode(mode) {
  if (isValidResetMode(mode)) {
    return mode;
  }
  switch (typeof mode) {
    case "string":
    case "undefined":
      return "soft";
  }
  return;
}
function isValidResetMode(mode) {
  return typeof mode === "string" && validResetModes.includes(mode);
}
var ResetMode;
var validResetModes;
var init_reset = __esm2({
  "src/lib/tasks/reset.ts"() {
    "use strict";
    init_utils();
    init_task();
    ResetMode = /* @__PURE__ */ ((ResetMode2) => {
      ResetMode2["MIXED"] = "mixed";
      ResetMode2["SOFT"] = "soft";
      ResetMode2["HARD"] = "hard";
      ResetMode2["MERGE"] = "merge";
      ResetMode2["KEEP"] = "keep";
      return ResetMode2;
    })(ResetMode || {});
    validResetModes = asStringArray(Object.values(ResetMode));
  }
});
function createLog() {
  return (0, import_debug.default)("simple-git");
}
function prefixedLogger(to, prefix, forward) {
  if (!prefix || !String(prefix).replace(/\s*/, "")) {
    return !forward ? to : (message, ...args) => {
      to(message, ...args);
      forward(message, ...args);
    };
  }
  return (message, ...args) => {
    to(`%s ${message}`, prefix, ...args);
    if (forward) {
      forward(message, ...args);
    }
  };
}
function childLoggerName(name, childDebugger, { namespace: parentNamespace }) {
  if (typeof name === "string") {
    return name;
  }
  const childNamespace = childDebugger && childDebugger.namespace || "";
  if (childNamespace.startsWith(parentNamespace)) {
    return childNamespace.substr(parentNamespace.length + 1);
  }
  return childNamespace || parentNamespace;
}
function createLogger(label, verbose, initialStep, infoDebugger = createLog()) {
  const labelPrefix = label && `[${label}]` || "";
  const spawned = [];
  const debugDebugger = typeof verbose === "string" ? infoDebugger.extend(verbose) : verbose;
  const key = childLoggerName(filterType(verbose, filterString), debugDebugger, infoDebugger);
  return step(initialStep);
  function sibling(name, initial) {
    return append(
      spawned,
      createLogger(label, key.replace(/^[^:]+/, name), initial, infoDebugger)
    );
  }
  function step(phase) {
    const stepPrefix = phase && `[${phase}]` || "";
    const debug2 = debugDebugger && prefixedLogger(debugDebugger, stepPrefix) || NOOP;
    const info = prefixedLogger(infoDebugger, `${labelPrefix} ${stepPrefix}`, debug2);
    return Object.assign(debugDebugger ? debug2 : info, {
      label,
      sibling,
      info,
      step
    });
  }
}
var init_git_logger = __esm2({
  "src/lib/git-logger.ts"() {
    "use strict";
    init_utils();
    import_debug.default.formatters.L = (value) => String(filterHasLength(value) ? value.length : "-");
    import_debug.default.formatters.B = (value) => {
      if (Buffer.isBuffer(value)) {
        return value.toString("utf8");
      }
      return objectToString(value);
    };
  }
});
var TasksPendingQueue;
var init_tasks_pending_queue = __esm2({
  "src/lib/runners/tasks-pending-queue.ts"() {
    "use strict";
    init_git_error();
    init_git_logger();
    TasksPendingQueue = class _TasksPendingQueue {
      constructor(logLabel = "GitExecutor") {
        this.logLabel = logLabel;
        this._queue = /* @__PURE__ */ new Map();
      }
      withProgress(task) {
        return this._queue.get(task);
      }
      createProgress(task) {
        const name = _TasksPendingQueue.getName(task.commands[0]);
        const logger = createLogger(this.logLabel, name);
        return {
          task,
          logger,
          name
        };
      }
      push(task) {
        const progress = this.createProgress(task);
        progress.logger("Adding task to the queue, commands = %o", task.commands);
        this._queue.set(task, progress);
        return progress;
      }
      fatal(err) {
        for (const [task, { logger }] of Array.from(this._queue.entries())) {
          if (task === err.task) {
            logger.info(`Failed %o`, err);
            logger(
              `Fatal exception, any as-yet un-started tasks run through this executor will not be attempted`
            );
          } else {
            logger.info(
              `A fatal exception occurred in a previous task, the queue has been purged: %o`,
              err.message
            );
          }
          this.complete(task);
        }
        if (this._queue.size !== 0) {
          throw new Error(`Queue size should be zero after fatal: ${this._queue.size}`);
        }
      }
      complete(task) {
        const progress = this.withProgress(task);
        if (progress) {
          this._queue.delete(task);
        }
      }
      attempt(task) {
        const progress = this.withProgress(task);
        if (!progress) {
          throw new GitError(void 0, "TasksPendingQueue: attempt called for an unknown task");
        }
        progress.logger("Starting task");
        return progress;
      }
      static getName(name = "empty") {
        return `task:${name}:${++_TasksPendingQueue.counter}`;
      }
      static {
        this.counter = 0;
      }
    };
  }
});
function pluginContext(task, commands) {
  return {
    method: first(task.commands) || "",
    commands
  };
}
function onErrorReceived(target2, logger) {
  return (err) => {
    logger(`[ERROR] child process exception %o`, err);
    target2.push(Buffer.from(String(err.stack), "ascii"));
  };
}
function onDataReceived(target2, name, logger, output) {
  return (buffer) => {
    logger(`%s received %L bytes`, name, buffer);
    output(`%B`, buffer);
    target2.push(buffer);
  };
}
var GitExecutorChain;
var init_git_executor_chain = __esm2({
  "src/lib/runners/git-executor-chain.ts"() {
    "use strict";
    init_git_error();
    init_task();
    init_utils();
    init_tasks_pending_queue();
    GitExecutorChain = class {
      constructor(_executor, _scheduler, _plugins) {
        this._executor = _executor;
        this._scheduler = _scheduler;
        this._plugins = _plugins;
        this._chain = Promise.resolve();
        this._queue = new TasksPendingQueue();
      }
      get cwd() {
        return this._cwd || this._executor.cwd;
      }
      set cwd(cwd) {
        this._cwd = cwd;
      }
      get env() {
        return this._executor.env;
      }
      get outputHandler() {
        return this._executor.outputHandler;
      }
      chain() {
        return this;
      }
      push(task) {
        this._queue.push(task);
        return this._chain = this._chain.then(() => this.attemptTask(task));
      }
      async attemptTask(task) {
        const onScheduleComplete = await this._scheduler.next();
        const onQueueComplete = () => this._queue.complete(task);
        try {
          const { logger } = this._queue.attempt(task);
          return await (isEmptyTask(task) ? this.attemptEmptyTask(task, logger) : this.attemptRemoteTask(task, logger));
        } catch (e) {
          throw this.onFatalException(task, e);
        } finally {
          onQueueComplete();
          onScheduleComplete();
        }
      }
      onFatalException(task, e) {
        const gitError = e instanceof GitError ? Object.assign(e, { task }) : new GitError(task, e && String(e));
        this._chain = Promise.resolve();
        this._queue.fatal(gitError);
        return gitError;
      }
      async attemptRemoteTask(task, logger) {
        const binary = this._plugins.exec("spawn.binary", "", pluginContext(task, task.commands));
        const args = this._plugins.exec("spawn.args", [...task.commands], {
          ...pluginContext(task, task.commands),
          env: { ...this.env }
        });
        const raw = await this.gitResponse(
          task,
          binary,
          args,
          this.outputHandler,
          logger.step("SPAWN")
        );
        const outputStreams = await this.handleTaskData(task, args, raw, logger.step("HANDLE"));
        logger(`passing response to task's parser as a %s`, task.format);
        if (isBufferTask(task)) {
          return callTaskParser(task.parser, outputStreams);
        }
        return callTaskParser(task.parser, outputStreams.asStrings());
      }
      async attemptEmptyTask(task, logger) {
        logger(`empty task bypassing child process to call to task's parser`);
        return task.parser(this);
      }
      handleTaskData(task, args, result, logger) {
        const { exitCode, rejection, stdOut, stdErr } = result;
        return new Promise((done, fail) => {
          logger(`Preparing to handle process response exitCode=%d stdOut=`, exitCode);
          const { error } = this._plugins.exec(
            "task.error",
            { error: rejection },
            {
              ...pluginContext(task, args),
              ...result
            }
          );
          if (error && task.onError) {
            logger.info(`exitCode=%s handling with custom error handler`);
            return task.onError(
              result,
              error,
              (newStdOut) => {
                logger.info(`custom error handler treated as success`);
                logger(`custom error returned a %s`, objectToString(newStdOut));
                done(
                  new GitOutputStreams(
                    Array.isArray(newStdOut) ? Buffer.concat(newStdOut) : newStdOut,
                    Buffer.concat(stdErr)
                  )
                );
              },
              fail
            );
          }
          if (error) {
            logger.info(
              `handling as error: exitCode=%s stdErr=%s rejection=%o`,
              exitCode,
              stdErr.length,
              rejection
            );
            return fail(error);
          }
          logger.info(`retrieving task output complete`);
          done(new GitOutputStreams(Buffer.concat(stdOut), Buffer.concat(stdErr)));
        });
      }
      async gitResponse(task, command, args, outputHandler, logger) {
        const outputLogger = logger.sibling("output");
        const spawnOptions = this._plugins.exec(
          "spawn.options",
          {
            cwd: this.cwd,
            env: this.env,
            windowsHide: true
          },
          pluginContext(task, task.commands)
        );
        return new Promise((done) => {
          const stdOut = [];
          const stdErr = [];
          logger.info(`%s %o`, command, args);
          logger("%O", spawnOptions);
          let rejection = this._beforeSpawn(task, args);
          if (rejection) {
            return done({
              stdOut,
              stdErr,
              exitCode: 9901,
              rejection
            });
          }
          this._plugins.exec("spawn.before", void 0, {
            ...pluginContext(task, args),
            kill(reason) {
              rejection = reason || rejection;
            }
          });
          const spawned = spawn2(command, args, spawnOptions);
          spawned.stdout.on(
            "data",
            onDataReceived(stdOut, "stdOut", logger, outputLogger.step("stdOut"))
          );
          spawned.stderr.on(
            "data",
            onDataReceived(stdErr, "stdErr", logger, outputLogger.step("stdErr"))
          );
          spawned.on("error", onErrorReceived(stdErr, logger));
          if (outputHandler) {
            logger(`Passing child process stdOut/stdErr to custom outputHandler`);
            outputHandler(command, spawned.stdout, spawned.stderr, [...args]);
          }
          this._plugins.exec("spawn.after", void 0, {
            ...pluginContext(task, args),
            spawned,
            close(exitCode, reason) {
              done({
                stdOut,
                stdErr,
                exitCode,
                rejection: rejection || reason
              });
            },
            kill(reason) {
              if (spawned.killed) {
                return;
              }
              rejection = reason;
              spawned.kill("SIGINT");
            }
          });
        });
      }
      _beforeSpawn(task, args) {
        let rejection;
        this._plugins.exec("spawn.before", void 0, {
          ...pluginContext(task, args),
          kill(reason) {
            rejection = reason || rejection;
          }
        });
        return rejection;
      }
    };
  }
});
var git_executor_exports = {};
__export2(git_executor_exports, {
  GitExecutor: () => GitExecutor
});
var GitExecutor;
var init_git_executor = __esm2({
  "src/lib/runners/git-executor.ts"() {
    "use strict";
    init_git_executor_chain();
    GitExecutor = class {
      constructor(cwd, _scheduler, _plugins) {
        this.cwd = cwd;
        this._scheduler = _scheduler;
        this._plugins = _plugins;
        this._chain = new GitExecutorChain(this, this._scheduler, this._plugins);
      }
      chain() {
        return new GitExecutorChain(this, this._scheduler, this._plugins);
      }
      push(task) {
        return this._chain.push(task);
      }
    };
  }
});
function taskCallback(task, response, callback = NOOP) {
  const onSuccess = (data) => {
    callback(null, data);
  };
  const onError2 = (err) => {
    if (err?.task === task) {
      callback(
        err instanceof GitResponseError ? addDeprecationNoticeToError(err) : err,
        void 0
      );
    }
  };
  response.then(onSuccess, onError2);
}
function addDeprecationNoticeToError(err) {
  let log = (name) => {
    console.warn(
      `simple-git deprecation notice: accessing GitResponseError.${name} should be GitResponseError.git.${name}, this will no longer be available in version 3`
    );
    log = NOOP;
  };
  return Object.create(err, Object.getOwnPropertyNames(err.git).reduce(descriptorReducer, {}));
  function descriptorReducer(all, name) {
    if (name in err) {
      return all;
    }
    all[name] = {
      enumerable: false,
      configurable: false,
      get() {
        log(name);
        return err.git[name];
      }
    };
    return all;
  }
}
var init_task_callback = __esm2({
  "src/lib/task-callback.ts"() {
    "use strict";
    init_git_response_error();
    init_utils();
  }
});
function changeWorkingDirectoryTask(directory, root) {
  return adhocExecTask((instance) => {
    if (!folderExists(directory)) {
      throw new Error(`Git.cwd: cannot change to non-directory "${directory}"`);
    }
    return (root || instance).cwd = directory;
  });
}
var init_change_working_directory = __esm2({
  "src/lib/tasks/change-working-directory.ts"() {
    "use strict";
    init_utils();
    init_task();
  }
});
function checkoutTask(args) {
  const commands = ["checkout", ...args];
  if (commands[1] === "-b" && commands.includes("-B")) {
    commands[1] = remove(commands, "-B");
  }
  return straightThroughStringTask(commands);
}
function checkout_default() {
  return {
    checkout() {
      return this._runTask(
        checkoutTask(getTrailingOptions(arguments, 1)),
        trailingFunctionArgument(arguments)
      );
    },
    checkoutBranch(branchName, startPoint) {
      return this._runTask(
        checkoutTask(["-b", branchName, startPoint, ...getTrailingOptions(arguments)]),
        trailingFunctionArgument(arguments)
      );
    },
    checkoutLocalBranch(branchName) {
      return this._runTask(
        checkoutTask(["-b", branchName, ...getTrailingOptions(arguments)]),
        trailingFunctionArgument(arguments)
      );
    }
  };
}
var init_checkout = __esm2({
  "src/lib/tasks/checkout.ts"() {
    "use strict";
    init_utils();
    init_task();
  }
});
function countObjectsResponse() {
  return {
    count: 0,
    garbage: 0,
    inPack: 0,
    packs: 0,
    prunePackable: 0,
    size: 0,
    sizeGarbage: 0,
    sizePack: 0
  };
}
function count_objects_default() {
  return {
    countObjects() {
      return this._runTask({
        commands: ["count-objects", "--verbose"],
        format: "utf-8",
        parser(stdOut) {
          return parseStringResponse(countObjectsResponse(), [parser2], stdOut);
        }
      });
    }
  };
}
var parser2;
var init_count_objects = __esm2({
  "src/lib/tasks/count-objects.ts"() {
    "use strict";
    init_utils();
    parser2 = new LineParser(
      /([a-z-]+): (\d+)$/,
      (result, [key, value]) => {
        const property = asCamelCase(key);
        if (Object.hasOwn(result, property)) {
          result[property] = asNumber(value);
        }
      }
    );
  }
});
function parseCommitResult(stdOut) {
  const result = {
    author: null,
    branch: "",
    commit: "",
    root: false,
    summary: {
      changes: 0,
      insertions: 0,
      deletions: 0
    }
  };
  return parseStringResponse(result, parsers, stdOut);
}
var parsers;
var init_parse_commit = __esm2({
  "src/lib/parsers/parse-commit.ts"() {
    "use strict";
    init_utils();
    parsers = [
      new LineParser(/^\[([^\s]+)( \([^)]+\))? ([^\]]+)/, (result, [branch, root, commit]) => {
        result.branch = branch;
        result.commit = commit;
        result.root = !!root;
      }),
      new LineParser(/\s*Author:\s(.+)/i, (result, [author]) => {
        const parts = author.split("<");
        const email = parts.pop();
        if (!email || !email.includes("@")) {
          return;
        }
        result.author = {
          email: email.substr(0, email.length - 1),
          name: parts.join("<").trim()
        };
      }),
      new LineParser(
        /(\d+)[^,]*(?:,\s*(\d+)[^,]*)(?:,\s*(\d+))/g,
        (result, [changes, insertions, deletions]) => {
          result.summary.changes = parseInt(changes, 10) || 0;
          result.summary.insertions = parseInt(insertions, 10) || 0;
          result.summary.deletions = parseInt(deletions, 10) || 0;
        }
      ),
      new LineParser(
        /^(\d+)[^,]*(?:,\s*(\d+)[^(]+\(([+-]))?/,
        (result, [changes, lines, direction]) => {
          result.summary.changes = parseInt(changes, 10) || 0;
          const count = parseInt(lines, 10) || 0;
          if (direction === "-") {
            result.summary.deletions = count;
          } else if (direction === "+") {
            result.summary.insertions = count;
          }
        }
      )
    ];
  }
});
function commitTask(message, files, customArgs) {
  const commands = [
    "-c",
    "core.abbrev=40",
    "commit",
    ...prefixedArray(message, "-m"),
    ...files,
    ...customArgs
  ];
  return {
    commands,
    format: "utf-8",
    parser: parseCommitResult
  };
}
function commit_default() {
  return {
    commit(message, ...rest) {
      const next = trailingFunctionArgument(arguments);
      const task = rejectDeprecatedSignatures(message) || commitTask(
        asArray(message),
        asArray(filterType(rest[0], filterStringOrStringArray, [])),
        [
          ...asStringArray(filterType(rest[1], filterArray, [])),
          ...getTrailingOptions(arguments, 0, true)
        ]
      );
      return this._runTask(task, next);
    }
  };
  function rejectDeprecatedSignatures(message) {
    return !filterStringOrStringArray(message) && configurationErrorTask(
      `git.commit: requires the commit message to be supplied as a string/string[]`
    );
  }
}
var init_commit = __esm2({
  "src/lib/tasks/commit.ts"() {
    "use strict";
    init_parse_commit();
    init_utils();
    init_task();
  }
});
function first_commit_default() {
  return {
    firstCommit() {
      return this._runTask(
        straightThroughStringTask(["rev-list", "--max-parents=0", "HEAD"], true),
        trailingFunctionArgument(arguments)
      );
    }
  };
}
var init_first_commit = __esm2({
  "src/lib/tasks/first-commit.ts"() {
    "use strict";
    init_utils();
    init_task();
  }
});
function hashObjectTask(filePath, write) {
  const commands = ["hash-object", filePath];
  if (write) {
    commands.push("-w");
  }
  return straightThroughStringTask(commands, true);
}
var init_hash_object = __esm2({
  "src/lib/tasks/hash-object.ts"() {
    "use strict";
    init_task();
  }
});
function parseInit(bare, path23, text) {
  const response = String(text).trim();
  let result;
  if (result = initResponseRegex.exec(response)) {
    return new InitSummary(bare, path23, false, result[1]);
  }
  if (result = reInitResponseRegex.exec(response)) {
    return new InitSummary(bare, path23, true, result[1]);
  }
  let gitDir = "";
  const tokens = response.split(" ");
  while (tokens.length) {
    const token = tokens.shift();
    if (token === "in") {
      gitDir = tokens.join(" ");
      break;
    }
  }
  return new InitSummary(bare, path23, /^re/i.test(response), gitDir);
}
var InitSummary;
var initResponseRegex;
var reInitResponseRegex;
var init_InitSummary = __esm2({
  "src/lib/responses/InitSummary.ts"() {
    "use strict";
    InitSummary = class {
      constructor(bare, path23, existing, gitDir) {
        this.bare = bare;
        this.path = path23;
        this.existing = existing;
        this.gitDir = gitDir;
      }
    };
    initResponseRegex = /^Init.+ repository in (.+)$/;
    reInitResponseRegex = /^Rein.+ in (.+)$/;
  }
});
function hasBareCommand(command) {
  return command.includes(bareCommand);
}
function initTask(bare = false, path23, customArgs) {
  const commands = ["init", ...customArgs];
  if (bare && !hasBareCommand(commands)) {
    commands.splice(1, 0, bareCommand);
  }
  return {
    commands,
    format: "utf-8",
    parser(text) {
      return parseInit(commands.includes("--bare"), path23, text);
    }
  };
}
var bareCommand;
var init_init = __esm2({
  "src/lib/tasks/init.ts"() {
    "use strict";
    init_InitSummary();
    bareCommand = "--bare";
  }
});
function logFormatFromCommand(customArgs) {
  for (let i2 = 0; i2 < customArgs.length; i2++) {
    const format = logFormatRegex.exec(customArgs[i2]);
    if (format) {
      return `--${format[1]}`;
    }
  }
  return "";
}
function isLogFormat(customArg) {
  return logFormatRegex.test(customArg);
}
var logFormatRegex;
var init_log_format = __esm2({
  "src/lib/args/log-format.ts"() {
    "use strict";
    logFormatRegex = /^--(stat|numstat|name-only|name-status)(=|$)/;
  }
});
var DiffSummary;
var init_DiffSummary = __esm2({
  "src/lib/responses/DiffSummary.ts"() {
    "use strict";
    DiffSummary = class {
      constructor() {
        this.changed = 0;
        this.deletions = 0;
        this.insertions = 0;
        this.files = [];
      }
    };
  }
});
function getDiffParser(format = "") {
  const parser4 = diffSummaryParsers[format];
  return (stdOut) => parseStringResponse(new DiffSummary(), parser4, stdOut, false);
}
var statParser;
var numStatParser;
var nameOnlyParser;
var nameStatusParser;
var diffSummaryParsers;
var init_parse_diff_summary = __esm2({
  "src/lib/parsers/parse-diff-summary.ts"() {
    "use strict";
    init_log_format();
    init_DiffSummary();
    init_diff_name_status();
    init_utils();
    statParser = [
      new LineParser(
        /^(.+)\s+\|\s+(\d+)(\s+[+\-]+)?$/,
        (result, [file, changes, alterations = ""]) => {
          result.files.push({
            file: file.trim(),
            changes: asNumber(changes),
            insertions: alterations.replace(/[^+]/g, "").length,
            deletions: alterations.replace(/[^-]/g, "").length,
            binary: false
          });
        }
      ),
      new LineParser(
        /^(.+) \|\s+Bin ([0-9.]+) -> ([0-9.]+) ([a-z]+)/,
        (result, [file, before, after]) => {
          result.files.push({
            file: file.trim(),
            before: asNumber(before),
            after: asNumber(after),
            binary: true
          });
        }
      ),
      new LineParser(
        /(\d+) files? changed\s*((?:, \d+ [^,]+){0,2})/,
        (result, [changed, summary]) => {
          const inserted = /(\d+) i/.exec(summary);
          const deleted = /(\d+) d/.exec(summary);
          result.changed = asNumber(changed);
          result.insertions = asNumber(inserted?.[1]);
          result.deletions = asNumber(deleted?.[1]);
        }
      )
    ];
    numStatParser = [
      new LineParser(
        /(\d+)\t(\d+)\t(.+)$/,
        (result, [changesInsert, changesDelete, file]) => {
          const insertions = asNumber(changesInsert);
          const deletions = asNumber(changesDelete);
          result.changed++;
          result.insertions += insertions;
          result.deletions += deletions;
          result.files.push({
            file,
            changes: insertions + deletions,
            insertions,
            deletions,
            binary: false
          });
        }
      ),
      new LineParser(/-\t-\t(.+)$/, (result, [file]) => {
        result.changed++;
        result.files.push({
          file,
          after: 0,
          before: 0,
          binary: true
        });
      })
    ];
    nameOnlyParser = [
      new LineParser(/(.+)$/, (result, [file]) => {
        result.changed++;
        result.files.push({
          file,
          changes: 0,
          insertions: 0,
          deletions: 0,
          binary: false
        });
      })
    ];
    nameStatusParser = [
      new LineParser(
        /([ACDMRTUXB])([0-9]{0,3})\t(.[^\t]*)(\t(.[^\t]*))?$/,
        (result, [status, similarity, from, _to, to]) => {
          result.changed++;
          result.files.push({
            file: to ?? from,
            changes: 0,
            insertions: 0,
            deletions: 0,
            binary: false,
            status: orVoid(isDiffNameStatus(status) && status),
            from: orVoid(!!to && from !== to && from),
            similarity: asNumber(similarity)
          });
        }
      )
    ];
    diffSummaryParsers = {
      [
        ""
        /* NONE */
      ]: statParser,
      [
        "--stat"
        /* STAT */
      ]: statParser,
      [
        "--numstat"
        /* NUM_STAT */
      ]: numStatParser,
      [
        "--name-status"
        /* NAME_STATUS */
      ]: nameStatusParser,
      [
        "--name-only"
        /* NAME_ONLY */
      ]: nameOnlyParser
    };
  }
});
function lineBuilder(tokens, fields) {
  return fields.reduce(
    (line, field, index) => {
      line[field] = tokens[index] || "";
      return line;
    },
    /* @__PURE__ */ Object.create({ diff: null })
  );
}
function createListLogSummaryParser(splitter = SPLITTER, fields = defaultFieldNames, logFormat = "") {
  const parseDiffResult = getDiffParser(logFormat);
  return function(stdOut) {
    const all = toLinesWithContent(
      stdOut.trim(),
      false,
      START_BOUNDARY
    ).map(function(item) {
      const lineDetail = item.split(COMMIT_BOUNDARY);
      const listLogLine = lineBuilder(lineDetail[0].split(splitter), fields);
      if (lineDetail.length > 1 && !!lineDetail[1].trim()) {
        listLogLine.diff = parseDiffResult(lineDetail[1]);
      }
      return listLogLine;
    });
    return {
      all,
      latest: all.length && all[0] || null,
      total: all.length
    };
  };
}
var START_BOUNDARY;
var COMMIT_BOUNDARY;
var SPLITTER;
var defaultFieldNames;
var init_parse_list_log_summary = __esm2({
  "src/lib/parsers/parse-list-log-summary.ts"() {
    "use strict";
    init_utils();
    init_parse_diff_summary();
    init_log_format();
    START_BOUNDARY = "\xF2\xF2\xF2\xF2\xF2\xF2 ";
    COMMIT_BOUNDARY = " \xF2\xF2";
    SPLITTER = " \xF2 ";
    defaultFieldNames = ["hash", "date", "message", "refs", "author_name", "author_email"];
  }
});
var diff_exports = {};
__export2(diff_exports, {
  diffSummaryTask: () => diffSummaryTask,
  validateLogFormatConfig: () => validateLogFormatConfig
});
function diffSummaryTask(customArgs) {
  let logFormat = logFormatFromCommand(customArgs);
  const commands = ["diff"];
  if (logFormat === "") {
    logFormat = "--stat";
    commands.push("--stat=4096");
  }
  commands.push(...customArgs);
  return validateLogFormatConfig(commands) || {
    commands,
    format: "utf-8",
    parser: getDiffParser(logFormat)
  };
}
function validateLogFormatConfig(customArgs) {
  const flags = customArgs.filter(isLogFormat);
  if (flags.length > 1) {
    return configurationErrorTask(
      `Summary flags are mutually exclusive - pick one of ${flags.join(",")}`
    );
  }
  if (flags.length && customArgs.includes("-z")) {
    return configurationErrorTask(
      `Summary flag ${flags} parsing is not compatible with null termination option '-z'`
    );
  }
}
var init_diff = __esm2({
  "src/lib/tasks/diff.ts"() {
    "use strict";
    init_log_format();
    init_parse_diff_summary();
    init_task();
  }
});
function prettyFormat(format, splitter) {
  const fields = [];
  const formatStr = [];
  Object.keys(format).forEach((field) => {
    fields.push(field);
    formatStr.push(String(format[field]));
  });
  return [fields, formatStr.join(splitter)];
}
function userOptions(input) {
  return Object.keys(input).reduce((out, key) => {
    if (!(key in excludeOptions)) {
      out[key] = input[key];
    }
    return out;
  }, {});
}
function parseLogOptions(opt = {}, customArgs = []) {
  const splitter = filterType(opt.splitter, filterString, SPLITTER);
  const format = filterPlainObject(opt.format) ? opt.format : {
    hash: "%H",
    date: opt.strictDate === false ? "%ai" : "%aI",
    message: "%s",
    refs: "%D",
    body: opt.multiLine ? "%B" : "%b",
    author_name: opt.mailMap !== false ? "%aN" : "%an",
    author_email: opt.mailMap !== false ? "%aE" : "%ae"
  };
  const [fields, formatStr] = prettyFormat(format, splitter);
  const suffix = [];
  const command = [
    `--pretty=format:${START_BOUNDARY}${formatStr}${COMMIT_BOUNDARY}`,
    ...customArgs
  ];
  const maxCount = opt.n || opt["max-count"] || opt.maxCount;
  if (maxCount) {
    command.push(`--max-count=${maxCount}`);
  }
  if (opt.from || opt.to) {
    const rangeOperator = opt.symmetric !== false ? "..." : "..";
    suffix.push(`${opt.from || ""}${rangeOperator}${opt.to || ""}`);
  }
  if (filterString(opt.file)) {
    command.push("--follow", c(opt.file));
  }
  appendTaskOptions(userOptions(opt), command);
  return {
    fields,
    splitter,
    commands: [...command, ...suffix]
  };
}
function logTask(splitter, fields, customArgs) {
  const parser4 = createListLogSummaryParser(splitter, fields, logFormatFromCommand(customArgs));
  return {
    commands: ["log", ...customArgs],
    format: "utf-8",
    parser: parser4
  };
}
function log_default() {
  return {
    log(...rest) {
      const next = trailingFunctionArgument(arguments);
      const options = parseLogOptions(
        trailingOptionsArgument(arguments),
        asStringArray(filterType(arguments[0], filterArray, []))
      );
      const task = rejectDeprecatedSignatures(...rest) || validateLogFormatConfig(options.commands) || createLogTask(options);
      return this._runTask(task, next);
    }
  };
  function createLogTask(options) {
    return logTask(options.splitter, options.fields, options.commands);
  }
  function rejectDeprecatedSignatures(from, to) {
    return filterString(from) && filterString(to) && configurationErrorTask(
      `git.log(string, string) should be replaced with git.log({ from: string, to: string })`
    );
  }
}
var excludeOptions;
var init_log = __esm2({
  "src/lib/tasks/log.ts"() {
    "use strict";
    init_log_format();
    init_parse_list_log_summary();
    init_utils();
    init_task();
    init_diff();
    excludeOptions = /* @__PURE__ */ ((excludeOptions2) => {
      excludeOptions2[excludeOptions2["--pretty"] = 0] = "--pretty";
      excludeOptions2[excludeOptions2["max-count"] = 1] = "max-count";
      excludeOptions2[excludeOptions2["maxCount"] = 2] = "maxCount";
      excludeOptions2[excludeOptions2["n"] = 3] = "n";
      excludeOptions2[excludeOptions2["file"] = 4] = "file";
      excludeOptions2[excludeOptions2["format"] = 5] = "format";
      excludeOptions2[excludeOptions2["from"] = 6] = "from";
      excludeOptions2[excludeOptions2["to"] = 7] = "to";
      excludeOptions2[excludeOptions2["splitter"] = 8] = "splitter";
      excludeOptions2[excludeOptions2["symmetric"] = 9] = "symmetric";
      excludeOptions2[excludeOptions2["mailMap"] = 10] = "mailMap";
      excludeOptions2[excludeOptions2["multiLine"] = 11] = "multiLine";
      excludeOptions2[excludeOptions2["strictDate"] = 12] = "strictDate";
      return excludeOptions2;
    })(excludeOptions || {});
  }
});
var MergeSummaryConflict;
var MergeSummaryDetail;
var init_MergeSummary = __esm2({
  "src/lib/responses/MergeSummary.ts"() {
    "use strict";
    MergeSummaryConflict = class {
      constructor(reason, file = null, meta) {
        this.reason = reason;
        this.file = file;
        this.meta = meta;
      }
      toString() {
        return `${this.file}:${this.reason}`;
      }
    };
    MergeSummaryDetail = class {
      constructor() {
        this.conflicts = [];
        this.merges = [];
        this.result = "success";
      }
      get failed() {
        return this.conflicts.length > 0;
      }
      get reason() {
        return this.result;
      }
      toString() {
        if (this.conflicts.length) {
          return `CONFLICTS: ${this.conflicts.join(", ")}`;
        }
        return "OK";
      }
    };
  }
});
var PullSummary;
var PullFailedSummary;
var init_PullSummary = __esm2({
  "src/lib/responses/PullSummary.ts"() {
    "use strict";
    PullSummary = class {
      constructor() {
        this.remoteMessages = {
          all: []
        };
        this.created = [];
        this.deleted = [];
        this.files = [];
        this.deletions = {};
        this.insertions = {};
        this.summary = {
          changes: 0,
          deletions: 0,
          insertions: 0
        };
      }
    };
    PullFailedSummary = class {
      constructor() {
        this.remote = "";
        this.hash = {
          local: "",
          remote: ""
        };
        this.branch = {
          local: "",
          remote: ""
        };
        this.message = "";
      }
      toString() {
        return this.message;
      }
    };
  }
});
function objectEnumerationResult(remoteMessages) {
  return remoteMessages.objects = remoteMessages.objects || {
    compressing: 0,
    counting: 0,
    enumerating: 0,
    packReused: 0,
    reused: { count: 0, delta: 0 },
    total: { count: 0, delta: 0 }
  };
}
function asObjectCount(source) {
  const count = /^\s*(\d+)/.exec(source);
  const delta = /delta (\d+)/i.exec(source);
  return {
    count: asNumber(count && count[1] || "0"),
    delta: asNumber(delta && delta[1] || "0")
  };
}
var remoteMessagesObjectParsers;
var init_parse_remote_objects = __esm2({
  "src/lib/parsers/parse-remote-objects.ts"() {
    "use strict";
    init_utils();
    remoteMessagesObjectParsers = [
      new RemoteLineParser(
        /^remote:\s*(enumerating|counting|compressing) objects: (\d+),/i,
        (result, [action, count]) => {
          const key = action.toLowerCase();
          const enumeration = objectEnumerationResult(result.remoteMessages);
          Object.assign(enumeration, { [key]: asNumber(count) });
        }
      ),
      new RemoteLineParser(
        /^remote:\s*(enumerating|counting|compressing) objects: \d+% \(\d+\/(\d+)\),/i,
        (result, [action, count]) => {
          const key = action.toLowerCase();
          const enumeration = objectEnumerationResult(result.remoteMessages);
          Object.assign(enumeration, { [key]: asNumber(count) });
        }
      ),
      new RemoteLineParser(
        /total ([^,]+), reused ([^,]+), pack-reused (\d+)/i,
        (result, [total, reused, packReused]) => {
          const objects = objectEnumerationResult(result.remoteMessages);
          objects.total = asObjectCount(total);
          objects.reused = asObjectCount(reused);
          objects.packReused = asNumber(packReused);
        }
      )
    ];
  }
});
function parseRemoteMessages(_stdOut, stdErr) {
  return parseStringResponse({ remoteMessages: new RemoteMessageSummary() }, parsers2, stdErr);
}
var parsers2;
var RemoteMessageSummary;
var init_parse_remote_messages = __esm2({
  "src/lib/parsers/parse-remote-messages.ts"() {
    "use strict";
    init_utils();
    init_parse_remote_objects();
    parsers2 = [
      new RemoteLineParser(/^remote:\s*(.+)$/, (result, [text]) => {
        result.remoteMessages.all.push(text.trim());
        return false;
      }),
      ...remoteMessagesObjectParsers,
      new RemoteLineParser(
        [/create a (?:pull|merge) request/i, /\s(https?:\/\/\S+)$/],
        (result, [pullRequestUrl]) => {
          result.remoteMessages.pullRequestUrl = pullRequestUrl;
        }
      ),
      new RemoteLineParser(
        [/found (\d+) vulnerabilities.+\(([^)]+)\)/i, /\s(https?:\/\/\S+)$/],
        (result, [count, summary, url]) => {
          result.remoteMessages.vulnerabilities = {
            count: asNumber(count),
            summary,
            url
          };
        }
      )
    ];
    RemoteMessageSummary = class {
      constructor() {
        this.all = [];
      }
    };
  }
});
function parsePullErrorResult(stdOut, stdErr) {
  const pullError = parseStringResponse(new PullFailedSummary(), errorParsers, [stdOut, stdErr]);
  return pullError.message && pullError;
}
var FILE_UPDATE_REGEX;
var SUMMARY_REGEX;
var ACTION_REGEX;
var parsers3;
var errorParsers;
var parsePullDetail;
var parsePullResult;
var init_parse_pull = __esm2({
  "src/lib/parsers/parse-pull.ts"() {
    "use strict";
    init_PullSummary();
    init_utils();
    init_parse_remote_messages();
    FILE_UPDATE_REGEX = /^\s*(.+?)\s+\|\s+\d+\s*(\+*)(-*)/;
    SUMMARY_REGEX = /(\d+)\D+((\d+)\D+\(\+\))?(\D+(\d+)\D+\(-\))?/;
    ACTION_REGEX = /^(create|delete) mode \d+ (.+)/;
    parsers3 = [
      new LineParser(FILE_UPDATE_REGEX, (result, [file, insertions, deletions]) => {
        result.files.push(file);
        if (insertions) {
          result.insertions[file] = insertions.length;
        }
        if (deletions) {
          result.deletions[file] = deletions.length;
        }
      }),
      new LineParser(SUMMARY_REGEX, (result, [changes, , insertions, , deletions]) => {
        if (insertions !== void 0 || deletions !== void 0) {
          result.summary.changes = +changes || 0;
          result.summary.insertions = +insertions || 0;
          result.summary.deletions = +deletions || 0;
          return true;
        }
        return false;
      }),
      new LineParser(ACTION_REGEX, (result, [action, file]) => {
        append(result.files, file);
        append(action === "create" ? result.created : result.deleted, file);
      })
    ];
    errorParsers = [
      new LineParser(/^from\s(.+)$/i, (result, [remote]) => void (result.remote = remote)),
      new LineParser(/^fatal:\s(.+)$/, (result, [message]) => void (result.message = message)),
      new LineParser(
        /([a-z0-9]+)\.\.([a-z0-9]+)\s+(\S+)\s+->\s+(\S+)$/,
        (result, [hashLocal, hashRemote, branchLocal, branchRemote]) => {
          result.branch.local = branchLocal;
          result.hash.local = hashLocal;
          result.branch.remote = branchRemote;
          result.hash.remote = hashRemote;
        }
      )
    ];
    parsePullDetail = (stdOut, stdErr) => {
      return parseStringResponse(new PullSummary(), parsers3, [stdOut, stdErr]);
    };
    parsePullResult = (stdOut, stdErr) => {
      return Object.assign(
        new PullSummary(),
        parsePullDetail(stdOut, stdErr),
        parseRemoteMessages(stdOut, stdErr)
      );
    };
  }
});
var parsers4;
var parseMergeResult;
var parseMergeDetail;
var init_parse_merge = __esm2({
  "src/lib/parsers/parse-merge.ts"() {
    "use strict";
    init_MergeSummary();
    init_utils();
    init_parse_pull();
    parsers4 = [
      new LineParser(/^Auto-merging\s+(.+)$/, (summary, [autoMerge]) => {
        summary.merges.push(autoMerge);
      }),
      new LineParser(/^CONFLICT\s+\((.+)\): Merge conflict in (.+)$/, (summary, [reason, file]) => {
        summary.conflicts.push(new MergeSummaryConflict(reason, file));
      }),
      new LineParser(
        /^CONFLICT\s+\((.+\/delete)\): (.+) deleted in (.+) and/,
        (summary, [reason, file, deleteRef]) => {
          summary.conflicts.push(new MergeSummaryConflict(reason, file, { deleteRef }));
        }
      ),
      new LineParser(/^CONFLICT\s+\((.+)\):/, (summary, [reason]) => {
        summary.conflicts.push(new MergeSummaryConflict(reason, null));
      }),
      new LineParser(/^Automatic merge failed;\s+(.+)$/, (summary, [result]) => {
        summary.result = result;
      })
    ];
    parseMergeResult = (stdOut, stdErr) => {
      return Object.assign(parseMergeDetail(stdOut, stdErr), parsePullResult(stdOut, stdErr));
    };
    parseMergeDetail = (stdOut) => {
      return parseStringResponse(new MergeSummaryDetail(), parsers4, stdOut);
    };
  }
});
function mergeTask(customArgs) {
  if (!customArgs.length) {
    return configurationErrorTask("Git.merge requires at least one option");
  }
  return {
    commands: ["merge", ...customArgs],
    format: "utf-8",
    parser(stdOut, stdErr) {
      const merge = parseMergeResult(stdOut, stdErr);
      if (merge.failed) {
        throw new GitResponseError(merge);
      }
      return merge;
    }
  };
}
var init_merge = __esm2({
  "src/lib/tasks/merge.ts"() {
    "use strict";
    init_git_response_error();
    init_parse_merge();
    init_task();
  }
});
function pushResultPushedItem(local, remote, status) {
  const deleted = status.includes("deleted");
  const tag = status.includes("tag") || /^refs\/tags/.test(local);
  const alreadyUpdated = !status.includes("new");
  return {
    deleted,
    tag,
    branch: !tag,
    new: !alreadyUpdated,
    alreadyUpdated,
    local,
    remote
  };
}
var parsers5;
var parsePushResult;
var parsePushDetail;
var init_parse_push = __esm2({
  "src/lib/parsers/parse-push.ts"() {
    "use strict";
    init_utils();
    init_parse_remote_messages();
    parsers5 = [
      new LineParser(/^Pushing to (.+)$/, (result, [repo]) => {
        result.repo = repo;
      }),
      new LineParser(/^updating local tracking ref '(.+)'/, (result, [local]) => {
        result.ref = {
          ...result.ref || {},
          local
        };
      }),
      new LineParser(/^[=*-]\s+([^:]+):(\S+)\s+\[(.+)]$/, (result, [local, remote, type]) => {
        result.pushed.push(pushResultPushedItem(local, remote, type));
      }),
      new LineParser(
        /^Branch '([^']+)' set up to track remote branch '([^']+)' from '([^']+)'/,
        (result, [local, remote, remoteName]) => {
          result.branch = {
            ...result.branch || {},
            local,
            remote,
            remoteName
          };
        }
      ),
      new LineParser(
        /^([^:]+):(\S+)\s+([a-z0-9]+)\.\.([a-z0-9]+)$/,
        (result, [local, remote, from, to]) => {
          result.update = {
            head: {
              local,
              remote
            },
            hash: {
              from,
              to
            }
          };
        }
      )
    ];
    parsePushResult = (stdOut, stdErr) => {
      const pushDetail = parsePushDetail(stdOut, stdErr);
      const responseDetail = parseRemoteMessages(stdOut, stdErr);
      return {
        ...pushDetail,
        ...responseDetail
      };
    };
    parsePushDetail = (stdOut, stdErr) => {
      return parseStringResponse({ pushed: [] }, parsers5, [stdOut, stdErr]);
    };
  }
});
var push_exports = {};
__export2(push_exports, {
  pushTagsTask: () => pushTagsTask,
  pushTask: () => pushTask
});
function pushTagsTask(ref = {}, customArgs) {
  append(customArgs, "--tags");
  return pushTask(ref, customArgs);
}
function pushTask(ref = {}, customArgs) {
  const commands = ["push", ...customArgs];
  if (ref.branch) {
    commands.splice(1, 0, ref.branch);
  }
  if (ref.remote) {
    commands.splice(1, 0, ref.remote);
  }
  remove(commands, "-v");
  append(commands, "--verbose");
  append(commands, "--porcelain");
  return {
    commands,
    format: "utf-8",
    parser: parsePushResult
  };
}
var init_push = __esm2({
  "src/lib/tasks/push.ts"() {
    "use strict";
    init_parse_push();
    init_utils();
  }
});
function show_default() {
  return {
    showBuffer() {
      const commands = ["show", ...getTrailingOptions(arguments, 1)];
      if (!commands.includes("--binary")) {
        commands.splice(1, 0, "--binary");
      }
      return this._runTask(
        straightThroughBufferTask(commands),
        trailingFunctionArgument(arguments)
      );
    },
    show() {
      const commands = ["show", ...getTrailingOptions(arguments, 1)];
      return this._runTask(
        straightThroughStringTask(commands),
        trailingFunctionArgument(arguments)
      );
    }
  };
}
var init_show = __esm2({
  "src/lib/tasks/show.ts"() {
    "use strict";
    init_utils();
    init_task();
  }
});
var fromPathRegex;
var FileStatusSummary;
var init_FileStatusSummary = __esm2({
  "src/lib/responses/FileStatusSummary.ts"() {
    "use strict";
    fromPathRegex = /^(.+)\0(.+)$/;
    FileStatusSummary = class {
      constructor(path23, index, working_dir) {
        this.path = path23;
        this.index = index;
        this.working_dir = working_dir;
        if (index === "R" || working_dir === "R") {
          const detail = fromPathRegex.exec(path23) || [null, path23, path23];
          this.from = detail[2] || "";
          this.path = detail[1] || "";
        }
      }
    };
  }
});
function renamedFile(line) {
  const [to, from] = line.split(NULL);
  return {
    from: from || to,
    to
  };
}
function parser3(indexX, indexY, handler) {
  return [`${indexX}${indexY}`, handler];
}
function conflicts(indexX, ...indexY) {
  return indexY.map((y2) => parser3(indexX, y2, (result, file) => result.conflicted.push(file)));
}
function splitLine(result, lineStr) {
  const trimmed2 = lineStr.trim();
  switch (" ") {
    case trimmed2.charAt(2):
      return data(trimmed2.charAt(0), trimmed2.charAt(1), trimmed2.slice(3));
    case trimmed2.charAt(1):
      return data(" ", trimmed2.charAt(0), trimmed2.slice(2));
    default:
      return;
  }
  function data(index, workingDir, path23) {
    const raw = `${index}${workingDir}`;
    const handler = parsers6.get(raw);
    if (handler) {
      handler(result, path23);
    }
    if (raw !== "##" && raw !== "!!") {
      result.files.push(new FileStatusSummary(path23, index, workingDir));
    }
  }
}
var StatusSummary;
var parsers6;
var parseStatusSummary;
var init_StatusSummary = __esm2({
  "src/lib/responses/StatusSummary.ts"() {
    "use strict";
    init_utils();
    init_FileStatusSummary();
    StatusSummary = class {
      constructor() {
        this.not_added = [];
        this.conflicted = [];
        this.created = [];
        this.deleted = [];
        this.ignored = void 0;
        this.modified = [];
        this.renamed = [];
        this.files = [];
        this.staged = [];
        this.ahead = 0;
        this.behind = 0;
        this.current = null;
        this.tracking = null;
        this.detached = false;
        this.isClean = () => {
          return !this.files.length;
        };
      }
    };
    parsers6 = new Map([
      parser3(
        " ",
        "A",
        (result, file) => result.created.push(file)
      ),
      parser3(
        " ",
        "D",
        (result, file) => result.deleted.push(file)
      ),
      parser3(
        " ",
        "M",
        (result, file) => result.modified.push(file)
      ),
      parser3("A", " ", (result, file) => {
        result.created.push(file);
        result.staged.push(file);
      }),
      parser3("A", "M", (result, file) => {
        result.created.push(file);
        result.staged.push(file);
        result.modified.push(file);
      }),
      parser3("D", " ", (result, file) => {
        result.deleted.push(file);
        result.staged.push(file);
      }),
      parser3("M", " ", (result, file) => {
        result.modified.push(file);
        result.staged.push(file);
      }),
      parser3("M", "M", (result, file) => {
        result.modified.push(file);
        result.staged.push(file);
      }),
      parser3("R", " ", (result, file) => {
        result.renamed.push(renamedFile(file));
      }),
      parser3("R", "M", (result, file) => {
        const renamed = renamedFile(file);
        result.renamed.push(renamed);
        result.modified.push(renamed.to);
      }),
      parser3("!", "!", (_result, _file) => {
        (_result.ignored = _result.ignored || []).push(_file);
      }),
      parser3(
        "?",
        "?",
        (result, file) => result.not_added.push(file)
      ),
      ...conflicts(
        "A",
        "A",
        "U"
        /* UNMERGED */
      ),
      ...conflicts(
        "D",
        "D",
        "U"
        /* UNMERGED */
      ),
      ...conflicts(
        "U",
        "A",
        "D",
        "U"
        /* UNMERGED */
      ),
      [
        "##",
        (result, line) => {
          const aheadReg = /ahead (\d+)/;
          const behindReg = /behind (\d+)/;
          const currentReg = /^(.+?(?=(?:\.{3}|\s|$)))/;
          const trackingReg = /\.{3}(\S*)/;
          const onEmptyBranchReg = /\son\s(\S+?)(?=\.{3}|$)/;
          let regexResult = aheadReg.exec(line);
          result.ahead = regexResult && +regexResult[1] || 0;
          regexResult = behindReg.exec(line);
          result.behind = regexResult && +regexResult[1] || 0;
          regexResult = currentReg.exec(line);
          result.current = filterType(regexResult?.[1], filterString, null);
          regexResult = trackingReg.exec(line);
          result.tracking = filterType(regexResult?.[1], filterString, null);
          regexResult = onEmptyBranchReg.exec(line);
          if (regexResult) {
            result.current = filterType(regexResult?.[1], filterString, result.current);
          }
          result.detached = /\(no branch\)/.test(line);
        }
      ]
    ]);
    parseStatusSummary = function(text) {
      const lines = text.split(NULL);
      const status = new StatusSummary();
      for (let i2 = 0, l = lines.length; i2 < l; ) {
        let line = lines[i2++].trim();
        if (!line) {
          continue;
        }
        if (line.charAt(0) === "R") {
          line += NULL + (lines[i2++] || "");
        }
        splitLine(status, line);
      }
      return status;
    };
  }
});
function statusTask(customArgs) {
  const commands = [
    "status",
    "--porcelain",
    "-b",
    "-u",
    "--null",
    ...customArgs.filter((arg) => !ignoredOptions.includes(arg))
  ];
  return {
    format: "utf-8",
    commands,
    parser(text) {
      return parseStatusSummary(text);
    }
  };
}
var ignoredOptions;
var init_status = __esm2({
  "src/lib/tasks/status.ts"() {
    "use strict";
    init_StatusSummary();
    ignoredOptions = ["--null", "-z"];
  }
});
function versionResponse(major = 0, minor = 0, patch = 0, agent = "", installed = true) {
  return Object.defineProperty(
    {
      major,
      minor,
      patch,
      agent,
      installed
    },
    "toString",
    {
      value() {
        return `${this.major}.${this.minor}.${this.patch}`;
      },
      configurable: false,
      enumerable: false
    }
  );
}
function notInstalledResponse() {
  return versionResponse(0, 0, 0, "", false);
}
function version_default() {
  return {
    version() {
      return this._runTask({
        commands: ["--version"],
        format: "utf-8",
        parser: versionParser,
        onError(result, error, done, fail) {
          if (result.exitCode === -2) {
            return done(Buffer.from(NOT_INSTALLED));
          }
          fail(error);
        }
      });
    }
  };
}
function versionParser(stdOut) {
  if (stdOut === NOT_INSTALLED) {
    return notInstalledResponse();
  }
  return parseStringResponse(versionResponse(0, 0, 0, stdOut), parsers7, stdOut);
}
var NOT_INSTALLED;
var parsers7;
var init_version = __esm2({
  "src/lib/tasks/version.ts"() {
    "use strict";
    init_utils();
    NOT_INSTALLED = "installed=false";
    parsers7 = [
      new LineParser(
        /version (\d+)\.(\d+)\.(\d+)(?:\s*\((.+)\))?/,
        (result, [major, minor, patch, agent = ""]) => {
          Object.assign(
            result,
            versionResponse(asNumber(major), asNumber(minor), asNumber(patch), agent)
          );
        }
      ),
      new LineParser(
        /version (\d+)\.(\d+)\.(\D+)(.+)?$/,
        (result, [major, minor, patch, agent = ""]) => {
          Object.assign(result, versionResponse(asNumber(major), asNumber(minor), patch, agent));
        }
      )
    ];
  }
});
function createCloneTask(api, task, repoPath, ...args) {
  if (!filterString(repoPath)) {
    return configurationErrorTask(`git.${api}() requires a string 'repoPath'`);
  }
  return task(repoPath, filterType(args[0], filterString), getTrailingOptions(arguments));
}
function clone_default() {
  return {
    clone(repo, ...rest) {
      return this._runTask(
        createCloneTask("clone", cloneTask, filterType(repo, filterString), ...rest),
        trailingFunctionArgument(arguments)
      );
    },
    mirror(repo, ...rest) {
      return this._runTask(
        createCloneTask("mirror", cloneMirrorTask, filterType(repo, filterString), ...rest),
        trailingFunctionArgument(arguments)
      );
    }
  };
}
var cloneTask;
var cloneMirrorTask;
var init_clone = __esm2({
  "src/lib/tasks/clone.ts"() {
    "use strict";
    init_task();
    init_utils();
    cloneTask = (repo, directory, customArgs) => {
      const commands = ["clone", ...customArgs];
      filterString(repo) && commands.push(c(repo));
      filterString(directory) && commands.push(c(directory));
      return straightThroughStringTask(commands);
    };
    cloneMirrorTask = (repo, directory, customArgs) => {
      append(customArgs, "--mirror");
      return cloneTask(repo, directory, customArgs);
    };
  }
});
var simple_git_api_exports = {};
__export2(simple_git_api_exports, {
  SimpleGitApi: () => SimpleGitApi
});
var SimpleGitApi;
var init_simple_git_api = __esm2({
  "src/lib/simple-git-api.ts"() {
    "use strict";
    init_task_callback();
    init_change_working_directory();
    init_checkout();
    init_count_objects();
    init_commit();
    init_config();
    init_first_commit();
    init_grep();
    init_hash_object();
    init_init();
    init_log();
    init_merge();
    init_push();
    init_show();
    init_status();
    init_task();
    init_version();
    init_utils();
    init_clone();
    SimpleGitApi = class {
      constructor(_executor) {
        this._executor = _executor;
      }
      _runTask(task, then) {
        const chain = this._executor.chain();
        const promise = chain.push(task);
        if (then) {
          taskCallback(task, promise, then);
        }
        return Object.create(this, {
          then: { value: promise.then.bind(promise) },
          catch: { value: promise.catch.bind(promise) },
          _executor: { value: chain }
        });
      }
      add(files) {
        return this._runTask(
          straightThroughStringTask(["add", ...asArray(files)]),
          trailingFunctionArgument(arguments)
        );
      }
      cwd(directory) {
        const next = trailingFunctionArgument(arguments);
        if (typeof directory === "string") {
          return this._runTask(changeWorkingDirectoryTask(directory, this._executor), next);
        }
        if (typeof directory?.path === "string") {
          return this._runTask(
            changeWorkingDirectoryTask(
              directory.path,
              directory.root && this._executor || void 0
            ),
            next
          );
        }
        return this._runTask(
          configurationErrorTask("Git.cwd: workingDirectory must be supplied as a string"),
          next
        );
      }
      hashObject(path23, write) {
        return this._runTask(
          hashObjectTask(path23, write === true),
          trailingFunctionArgument(arguments)
        );
      }
      init(bare) {
        return this._runTask(
          initTask(bare === true, this._executor.cwd, getTrailingOptions(arguments)),
          trailingFunctionArgument(arguments)
        );
      }
      merge() {
        return this._runTask(
          mergeTask(getTrailingOptions(arguments)),
          trailingFunctionArgument(arguments)
        );
      }
      mergeFromTo(remote, branch) {
        if (!(filterString(remote) && filterString(branch))) {
          return this._runTask(
            configurationErrorTask(
              `Git.mergeFromTo requires that the 'remote' and 'branch' arguments are supplied as strings`
            )
          );
        }
        return this._runTask(
          mergeTask([remote, branch, ...getTrailingOptions(arguments)]),
          trailingFunctionArgument(arguments, false)
        );
      }
      outputHandler(handler) {
        this._executor.outputHandler = handler;
        return this;
      }
      push() {
        const task = pushTask(
          {
            remote: filterType(arguments[0], filterString),
            branch: filterType(arguments[1], filterString)
          },
          getTrailingOptions(arguments)
        );
        return this._runTask(task, trailingFunctionArgument(arguments));
      }
      stash() {
        return this._runTask(
          straightThroughStringTask(["stash", ...getTrailingOptions(arguments)]),
          trailingFunctionArgument(arguments)
        );
      }
      status() {
        return this._runTask(
          statusTask(getTrailingOptions(arguments)),
          trailingFunctionArgument(arguments)
        );
      }
    };
    Object.assign(
      SimpleGitApi.prototype,
      checkout_default(),
      clone_default(),
      commit_default(),
      config_default(),
      count_objects_default(),
      first_commit_default(),
      grep_default(),
      log_default(),
      show_default(),
      version_default()
    );
  }
});
var scheduler_exports = {};
__export2(scheduler_exports, {
  Scheduler: () => Scheduler
});
var createScheduledTask;
var Scheduler;
var init_scheduler = __esm2({
  "src/lib/runners/scheduler.ts"() {
    "use strict";
    init_utils();
    init_git_logger();
    createScheduledTask = /* @__PURE__ */ (() => {
      let id = 0;
      return () => {
        id++;
        const { promise, done } = (0, import_promise_deferred.createDeferred)();
        return {
          promise,
          done,
          id
        };
      };
    })();
    Scheduler = class {
      constructor(concurrency = 2) {
        this.concurrency = concurrency;
        this.logger = createLogger("", "scheduler");
        this.pending = [];
        this.running = [];
        this.logger(`Constructed, concurrency=%s`, concurrency);
      }
      schedule() {
        if (!this.pending.length || this.running.length >= this.concurrency) {
          this.logger(
            `Schedule attempt ignored, pending=%s running=%s concurrency=%s`,
            this.pending.length,
            this.running.length,
            this.concurrency
          );
          return;
        }
        const task = append(this.running, this.pending.shift());
        this.logger(`Attempting id=%s`, task.id);
        task.done(() => {
          this.logger(`Completing id=`, task.id);
          remove(this.running, task);
          this.schedule();
        });
      }
      next() {
        const { promise, id } = append(this.pending, createScheduledTask());
        this.logger(`Scheduling id=%s`, id);
        this.schedule();
        return promise;
      }
    };
  }
});
var apply_patch_exports = {};
__export2(apply_patch_exports, {
  applyPatchTask: () => applyPatchTask
});
function applyPatchTask(patches, customArgs) {
  return straightThroughStringTask(["apply", ...customArgs, ...patches]);
}
var init_apply_patch = __esm2({
  "src/lib/tasks/apply-patch.ts"() {
    "use strict";
    init_task();
  }
});
function branchDeletionSuccess(branch, hash) {
  return {
    branch,
    hash,
    success: true
  };
}
function branchDeletionFailure(branch) {
  return {
    branch,
    hash: null,
    success: false
  };
}
var BranchDeletionBatch;
var init_BranchDeleteSummary = __esm2({
  "src/lib/responses/BranchDeleteSummary.ts"() {
    "use strict";
    BranchDeletionBatch = class {
      constructor() {
        this.all = [];
        this.branches = {};
        this.errors = [];
      }
      get success() {
        return !this.errors.length;
      }
    };
  }
});
function hasBranchDeletionError(data, processExitCode) {
  return processExitCode === 1 && deleteErrorRegex.test(data);
}
var deleteSuccessRegex;
var deleteErrorRegex;
var parsers8;
var parseBranchDeletions;
var init_parse_branch_delete = __esm2({
  "src/lib/parsers/parse-branch-delete.ts"() {
    "use strict";
    init_BranchDeleteSummary();
    init_utils();
    deleteSuccessRegex = /(\S+)\s+\(\S+\s([^)]+)\)/;
    deleteErrorRegex = /^error[^']+'([^']+)'/m;
    parsers8 = [
      new LineParser(deleteSuccessRegex, (result, [branch, hash]) => {
        const deletion = branchDeletionSuccess(branch, hash);
        result.all.push(deletion);
        result.branches[branch] = deletion;
      }),
      new LineParser(deleteErrorRegex, (result, [branch]) => {
        const deletion = branchDeletionFailure(branch);
        result.errors.push(deletion);
        result.all.push(deletion);
        result.branches[branch] = deletion;
      })
    ];
    parseBranchDeletions = (stdOut, stdErr) => {
      return parseStringResponse(new BranchDeletionBatch(), parsers8, [stdOut, stdErr]);
    };
  }
});
var BranchSummaryResult;
var init_BranchSummary = __esm2({
  "src/lib/responses/BranchSummary.ts"() {
    "use strict";
    BranchSummaryResult = class {
      constructor() {
        this.all = [];
        this.branches = {};
        this.current = "";
        this.detached = false;
      }
      push(status, detached, name, commit, label) {
        if (status === "*") {
          this.detached = detached;
          this.current = name;
        }
        this.all.push(name);
        this.branches[name] = {
          current: status === "*",
          linkedWorkTree: status === "+",
          name,
          commit,
          label
        };
      }
    };
  }
});
function branchStatus(input) {
  return input ? input.charAt(0) : "";
}
function parseBranchSummary(stdOut, currentOnly = false) {
  return parseStringResponse(
    new BranchSummaryResult(),
    currentOnly ? [currentBranchParser] : parsers9,
    stdOut
  );
}
var parsers9;
var currentBranchParser;
var init_parse_branch = __esm2({
  "src/lib/parsers/parse-branch.ts"() {
    "use strict";
    init_BranchSummary();
    init_utils();
    parsers9 = [
      new LineParser(
        /^([*+]\s)?\((?:HEAD )?detached (?:from|at) (\S+)\)\s+([a-z0-9]+)\s(.*)$/,
        (result, [current, name, commit, label]) => {
          result.push(branchStatus(current), true, name, commit, label);
        }
      ),
      new LineParser(
        /^([*+]\s)?(\S+)\s+([a-z0-9]+)\s?(.*)$/s,
        (result, [current, name, commit, label]) => {
          result.push(branchStatus(current), false, name, commit, label);
        }
      )
    ];
    currentBranchParser = new LineParser(/^(\S+)$/s, (result, [name]) => {
      result.push("*", false, name, "", "");
    });
  }
});
var branch_exports = {};
__export2(branch_exports, {
  branchLocalTask: () => branchLocalTask,
  branchTask: () => branchTask,
  containsDeleteBranchCommand: () => containsDeleteBranchCommand,
  deleteBranchTask: () => deleteBranchTask,
  deleteBranchesTask: () => deleteBranchesTask
});
function containsDeleteBranchCommand(commands) {
  const deleteCommands = ["-d", "-D", "--delete"];
  return commands.some((command) => deleteCommands.includes(command));
}
function branchTask(customArgs) {
  const isDelete = containsDeleteBranchCommand(customArgs);
  const isCurrentOnly = customArgs.includes("--show-current");
  const commands = ["branch", ...customArgs];
  if (commands.length === 1) {
    commands.push("-a");
  }
  if (!commands.includes("-v")) {
    commands.splice(1, 0, "-v");
  }
  return {
    format: "utf-8",
    commands,
    parser(stdOut, stdErr) {
      if (isDelete) {
        return parseBranchDeletions(stdOut, stdErr).all[0];
      }
      return parseBranchSummary(stdOut, isCurrentOnly);
    }
  };
}
function branchLocalTask() {
  return {
    format: "utf-8",
    commands: ["branch", "-v"],
    parser(stdOut) {
      return parseBranchSummary(stdOut);
    }
  };
}
function deleteBranchesTask(branches, forceDelete = false) {
  return {
    format: "utf-8",
    commands: ["branch", "-v", forceDelete ? "-D" : "-d", ...branches],
    parser(stdOut, stdErr) {
      return parseBranchDeletions(stdOut, stdErr);
    },
    onError({ exitCode, stdOut }, error, done, fail) {
      if (!hasBranchDeletionError(String(error), exitCode)) {
        return fail(error);
      }
      done(stdOut);
    }
  };
}
function deleteBranchTask(branch, forceDelete = false) {
  const task = {
    format: "utf-8",
    commands: ["branch", "-v", forceDelete ? "-D" : "-d", branch],
    parser(stdOut, stdErr) {
      return parseBranchDeletions(stdOut, stdErr).branches[branch];
    },
    onError({ exitCode, stdErr, stdOut }, error, _2, fail) {
      if (!hasBranchDeletionError(String(error), exitCode)) {
        return fail(error);
      }
      throw new GitResponseError(
        task.parser(bufferToString(stdOut), bufferToString(stdErr)),
        String(error)
      );
    }
  };
  return task;
}
var init_branch = __esm2({
  "src/lib/tasks/branch.ts"() {
    "use strict";
    init_git_response_error();
    init_parse_branch_delete();
    init_parse_branch();
    init_utils();
  }
});
function toPath(input) {
  const path23 = input.trim().replace(/^["']|["']$/g, "");
  return path23 && normalize(path23);
}
var parseCheckIgnore;
var init_CheckIgnore = __esm2({
  "src/lib/responses/CheckIgnore.ts"() {
    "use strict";
    parseCheckIgnore = (text) => {
      return text.split(/\n/g).map(toPath).filter(Boolean);
    };
  }
});
var check_ignore_exports = {};
__export2(check_ignore_exports, {
  checkIgnoreTask: () => checkIgnoreTask
});
function checkIgnoreTask(paths) {
  return {
    commands: ["check-ignore", ...paths],
    format: "utf-8",
    parser: parseCheckIgnore
  };
}
var init_check_ignore = __esm2({
  "src/lib/tasks/check-ignore.ts"() {
    "use strict";
    init_CheckIgnore();
  }
});
function parseFetchResult(stdOut, stdErr) {
  const result = {
    raw: stdOut,
    remote: null,
    branches: [],
    tags: [],
    updated: [],
    deleted: []
  };
  return parseStringResponse(result, parsers10, [stdOut, stdErr]);
}
var parsers10;
var init_parse_fetch = __esm2({
  "src/lib/parsers/parse-fetch.ts"() {
    "use strict";
    init_utils();
    parsers10 = [
      new LineParser(/From (.+)$/, (result, [remote]) => {
        result.remote = remote;
      }),
      new LineParser(/\* \[new branch]\s+(\S+)\s*-> (.+)$/, (result, [name, tracking]) => {
        result.branches.push({
          name,
          tracking
        });
      }),
      new LineParser(/\* \[new tag]\s+(\S+)\s*-> (.+)$/, (result, [name, tracking]) => {
        result.tags.push({
          name,
          tracking
        });
      }),
      new LineParser(/- \[deleted]\s+\S+\s*-> (.+)$/, (result, [tracking]) => {
        result.deleted.push({
          tracking
        });
      }),
      new LineParser(
        /\s*([^.]+)\.\.(\S+)\s+(\S+)\s*-> (.+)$/,
        (result, [from, to, name, tracking]) => {
          result.updated.push({
            name,
            tracking,
            to,
            from
          });
        }
      )
    ];
  }
});
var fetch_exports = {};
__export2(fetch_exports, {
  fetchTask: () => fetchTask
});
function disallowedCommand(command) {
  return /^--upload-pack(=|$)/.test(command);
}
function fetchTask(remote, branch, customArgs) {
  const commands = ["fetch", ...customArgs];
  if (remote && branch) {
    commands.push(remote, branch);
  }
  const banned = commands.find(disallowedCommand);
  if (banned) {
    return configurationErrorTask(`git.fetch: potential exploit argument blocked.`);
  }
  return {
    commands,
    format: "utf-8",
    parser: parseFetchResult
  };
}
var init_fetch = __esm2({
  "src/lib/tasks/fetch.ts"() {
    "use strict";
    init_parse_fetch();
    init_task();
  }
});
function parseMoveResult(stdOut) {
  return parseStringResponse({ moves: [] }, parsers11, stdOut);
}
var parsers11;
var init_parse_move = __esm2({
  "src/lib/parsers/parse-move.ts"() {
    "use strict";
    init_utils();
    parsers11 = [
      new LineParser(/^Renaming (.+) to (.+)$/, (result, [from, to]) => {
        result.moves.push({ from, to });
      })
    ];
  }
});
var move_exports = {};
__export2(move_exports, {
  moveTask: () => moveTask
});
function moveTask(from, to) {
  return {
    commands: ["mv", "-v", ...asArray(from), to],
    format: "utf-8",
    parser: parseMoveResult
  };
}
var init_move = __esm2({
  "src/lib/tasks/move.ts"() {
    "use strict";
    init_parse_move();
    init_utils();
  }
});
var pull_exports = {};
__export2(pull_exports, {
  pullTask: () => pullTask
});
function pullTask(remote, branch, customArgs) {
  const commands = ["pull", ...customArgs];
  if (remote && branch) {
    commands.splice(1, 0, remote, branch);
  }
  return {
    commands,
    format: "utf-8",
    parser(stdOut, stdErr) {
      return parsePullResult(stdOut, stdErr);
    },
    onError(result, _error, _done, fail) {
      const pullError = parsePullErrorResult(
        bufferToString(result.stdOut),
        bufferToString(result.stdErr)
      );
      if (pullError) {
        return fail(new GitResponseError(pullError));
      }
      fail(_error);
    }
  };
}
var init_pull = __esm2({
  "src/lib/tasks/pull.ts"() {
    "use strict";
    init_git_response_error();
    init_parse_pull();
    init_utils();
  }
});
function parseGetRemotes(text) {
  const remotes = {};
  forEach(text, ([name]) => remotes[name] = { name });
  return Object.values(remotes);
}
function parseGetRemotesVerbose(text) {
  const remotes = {};
  forEach(text, ([name, url, purpose]) => {
    if (!Object.hasOwn(remotes, name)) {
      remotes[name] = {
        name,
        refs: { fetch: "", push: "" }
      };
    }
    if (purpose && url) {
      remotes[name].refs[purpose.replace(/[^a-z]/g, "")] = url;
    }
  });
  return Object.values(remotes);
}
function forEach(text, handler) {
  forEachLineWithContent(text, (line) => handler(line.split(/\s+/)));
}
var init_GetRemoteSummary = __esm2({
  "src/lib/responses/GetRemoteSummary.ts"() {
    "use strict";
    init_utils();
  }
});
var remote_exports = {};
__export2(remote_exports, {
  addRemoteTask: () => addRemoteTask,
  getRemotesTask: () => getRemotesTask,
  listRemotesTask: () => listRemotesTask,
  remoteTask: () => remoteTask,
  removeRemoteTask: () => removeRemoteTask
});
function addRemoteTask(remoteName, remoteRepo, customArgs) {
  return straightThroughStringTask(["remote", "add", ...customArgs, remoteName, remoteRepo]);
}
function getRemotesTask(verbose) {
  const commands = ["remote"];
  if (verbose) {
    commands.push("-v");
  }
  return {
    commands,
    format: "utf-8",
    parser: verbose ? parseGetRemotesVerbose : parseGetRemotes
  };
}
function listRemotesTask(customArgs) {
  const commands = [...customArgs];
  if (commands[0] !== "ls-remote") {
    commands.unshift("ls-remote");
  }
  return straightThroughStringTask(commands);
}
function remoteTask(customArgs) {
  const commands = [...customArgs];
  if (commands[0] !== "remote") {
    commands.unshift("remote");
  }
  return straightThroughStringTask(commands);
}
function removeRemoteTask(remoteName) {
  return straightThroughStringTask(["remote", "remove", remoteName]);
}
var init_remote = __esm2({
  "src/lib/tasks/remote.ts"() {
    "use strict";
    init_GetRemoteSummary();
    init_task();
  }
});
var stash_list_exports = {};
__export2(stash_list_exports, {
  stashListTask: () => stashListTask
});
function stashListTask(opt = {}, customArgs) {
  const options = parseLogOptions(opt);
  const commands = ["stash", "list", ...options.commands, ...customArgs];
  const parser4 = createListLogSummaryParser(
    options.splitter,
    options.fields,
    logFormatFromCommand(commands)
  );
  return validateLogFormatConfig(commands) || {
    commands,
    format: "utf-8",
    parser: parser4
  };
}
var init_stash_list = __esm2({
  "src/lib/tasks/stash-list.ts"() {
    "use strict";
    init_log_format();
    init_parse_list_log_summary();
    init_diff();
    init_log();
  }
});
var sub_module_exports = {};
__export2(sub_module_exports, {
  addSubModuleTask: () => addSubModuleTask,
  initSubModuleTask: () => initSubModuleTask,
  subModuleTask: () => subModuleTask,
  updateSubModuleTask: () => updateSubModuleTask
});
function addSubModuleTask(repo, path23) {
  return subModuleTask(["add", repo, path23]);
}
function initSubModuleTask(customArgs) {
  return subModuleTask(["init", ...customArgs]);
}
function subModuleTask(customArgs) {
  const commands = [...customArgs];
  if (commands[0] !== "submodule") {
    commands.unshift("submodule");
  }
  return straightThroughStringTask(commands);
}
function updateSubModuleTask(customArgs) {
  return subModuleTask(["update", ...customArgs]);
}
var init_sub_module = __esm2({
  "src/lib/tasks/sub-module.ts"() {
    "use strict";
    init_task();
  }
});
function singleSorted(a, b2) {
  const aIsNum = Number.isNaN(a);
  const bIsNum = Number.isNaN(b2);
  if (aIsNum !== bIsNum) {
    return aIsNum ? 1 : -1;
  }
  return aIsNum ? sorted(a, b2) : 0;
}
function sorted(a, b2) {
  return a === b2 ? 0 : a > b2 ? 1 : -1;
}
function trimmed(input) {
  return input.trim();
}
function toNumber(input) {
  if (typeof input === "string") {
    return parseInt(input.replace(/^\D+/g, ""), 10) || 0;
  }
  return 0;
}
var TagList;
var parseTagList;
var init_TagList = __esm2({
  "src/lib/responses/TagList.ts"() {
    "use strict";
    TagList = class {
      constructor(all, latest) {
        this.all = all;
        this.latest = latest;
      }
    };
    parseTagList = function(data, customSort = false) {
      const tags = data.split("\n").map(trimmed).filter(Boolean);
      if (!customSort) {
        tags.sort(function(tagA, tagB) {
          const partsA = tagA.split(".");
          const partsB = tagB.split(".");
          if (partsA.length === 1 || partsB.length === 1) {
            return singleSorted(toNumber(partsA[0]), toNumber(partsB[0]));
          }
          for (let i2 = 0, l = Math.max(partsA.length, partsB.length); i2 < l; i2++) {
            const diff = sorted(toNumber(partsA[i2]), toNumber(partsB[i2]));
            if (diff) {
              return diff;
            }
          }
          return 0;
        });
      }
      const latest = customSort ? tags[0] : [...tags].reverse().find((tag) => tag.indexOf(".") >= 0);
      return new TagList(tags, latest);
    };
  }
});
var tag_exports = {};
__export2(tag_exports, {
  addAnnotatedTagTask: () => addAnnotatedTagTask,
  addTagTask: () => addTagTask,
  tagListTask: () => tagListTask
});
function tagListTask(customArgs = []) {
  const hasCustomSort = customArgs.some((option) => /^--sort=/.test(option));
  return {
    format: "utf-8",
    commands: ["tag", "-l", ...customArgs],
    parser(text) {
      return parseTagList(text, hasCustomSort);
    }
  };
}
function addTagTask(name) {
  return {
    format: "utf-8",
    commands: ["tag", name],
    parser() {
      return { name };
    }
  };
}
function addAnnotatedTagTask(name, tagMessage) {
  return {
    format: "utf-8",
    commands: ["tag", "-a", "-m", tagMessage, name],
    parser() {
      return { name };
    }
  };
}
var init_tag = __esm2({
  "src/lib/tasks/tag.ts"() {
    "use strict";
    init_TagList();
  }
});
var require_git = __commonJS2({
  "src/git.js"(exports, module) {
    "use strict";
    var { GitExecutor: GitExecutor2 } = (init_git_executor(), __toCommonJS2(git_executor_exports));
    var { SimpleGitApi: SimpleGitApi2 } = (init_simple_git_api(), __toCommonJS2(simple_git_api_exports));
    var { Scheduler: Scheduler2 } = (init_scheduler(), __toCommonJS2(scheduler_exports));
    var { adhocExecTask: adhocExecTask2, configurationErrorTask: configurationErrorTask2 } = (init_task(), __toCommonJS2(task_exports));
    var {
      asArray: asArray2,
      filterArray: filterArray2,
      filterPrimitives: filterPrimitives2,
      filterString: filterString2,
      filterStringOrStringArray: filterStringOrStringArray2,
      filterType: filterType2,
      getTrailingOptions: getTrailingOptions2,
      trailingFunctionArgument: trailingFunctionArgument2,
      trailingOptionsArgument: trailingOptionsArgument2
    } = (init_utils(), __toCommonJS2(utils_exports));
    var { applyPatchTask: applyPatchTask2 } = (init_apply_patch(), __toCommonJS2(apply_patch_exports));
    var {
      branchTask: branchTask2,
      branchLocalTask: branchLocalTask2,
      deleteBranchesTask: deleteBranchesTask2,
      deleteBranchTask: deleteBranchTask2
    } = (init_branch(), __toCommonJS2(branch_exports));
    var { checkIgnoreTask: checkIgnoreTask2 } = (init_check_ignore(), __toCommonJS2(check_ignore_exports));
    var { checkIsRepoTask: checkIsRepoTask2 } = (init_check_is_repo(), __toCommonJS2(check_is_repo_exports));
    var { cleanWithOptionsTask: cleanWithOptionsTask2, isCleanOptionsArray: isCleanOptionsArray2 } = (init_clean(), __toCommonJS2(clean_exports));
    var { diffSummaryTask: diffSummaryTask2 } = (init_diff(), __toCommonJS2(diff_exports));
    var { fetchTask: fetchTask2 } = (init_fetch(), __toCommonJS2(fetch_exports));
    var { moveTask: moveTask2 } = (init_move(), __toCommonJS2(move_exports));
    var { pullTask: pullTask2 } = (init_pull(), __toCommonJS2(pull_exports));
    var { pushTagsTask: pushTagsTask2 } = (init_push(), __toCommonJS2(push_exports));
    var {
      addRemoteTask: addRemoteTask2,
      getRemotesTask: getRemotesTask2,
      listRemotesTask: listRemotesTask2,
      remoteTask: remoteTask2,
      removeRemoteTask: removeRemoteTask2
    } = (init_remote(), __toCommonJS2(remote_exports));
    var { getResetMode: getResetMode2, resetTask: resetTask2 } = (init_reset(), __toCommonJS2(reset_exports));
    var { stashListTask: stashListTask2 } = (init_stash_list(), __toCommonJS2(stash_list_exports));
    var {
      addSubModuleTask: addSubModuleTask2,
      initSubModuleTask: initSubModuleTask2,
      subModuleTask: subModuleTask2,
      updateSubModuleTask: updateSubModuleTask2
    } = (init_sub_module(), __toCommonJS2(sub_module_exports));
    var { addAnnotatedTagTask: addAnnotatedTagTask2, addTagTask: addTagTask2, tagListTask: tagListTask2 } = (init_tag(), __toCommonJS2(tag_exports));
    var { straightThroughBufferTask: straightThroughBufferTask2, straightThroughStringTask: straightThroughStringTask2 } = (init_task(), __toCommonJS2(task_exports));
    function Git2(options, plugins) {
      this._plugins = plugins;
      this._executor = new GitExecutor2(
        options.baseDir,
        new Scheduler2(options.maxConcurrentProcesses),
        plugins
      );
      this._trimmed = options.trimmed;
    }
    (Git2.prototype = Object.create(SimpleGitApi2.prototype)).constructor = Git2;
    Git2.prototype.customBinary = function(command) {
      this._plugins.reconfigure("binary", command);
      return this;
    };
    Git2.prototype.env = function(name, value) {
      if (arguments.length === 1 && typeof name === "object") {
        this._executor.env = name;
      } else {
        (this._executor.env = this._executor.env || {})[name] = value;
      }
      return this;
    };
    Git2.prototype.stashList = function(options) {
      return this._runTask(
        stashListTask2(
          trailingOptionsArgument2(arguments) || {},
          filterArray2(options) && options || []
        ),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.mv = function(from, to) {
      return this._runTask(moveTask2(from, to), trailingFunctionArgument2(arguments));
    };
    Git2.prototype.checkoutLatestTag = function(then) {
      var git = this;
      return this.pull(function() {
        git.tags(function(err, tags) {
          git.checkout(tags.latest, then);
        });
      });
    };
    Git2.prototype.pull = function(remote, branch, options, then) {
      return this._runTask(
        pullTask2(
          filterType2(remote, filterString2),
          filterType2(branch, filterString2),
          getTrailingOptions2(arguments)
        ),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.fetch = function(remote, branch) {
      return this._runTask(
        fetchTask2(
          filterType2(remote, filterString2),
          filterType2(branch, filterString2),
          getTrailingOptions2(arguments)
        ),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.silent = function(silence) {
      return this._runTask(
        adhocExecTask2(
          () => console.warn(
            "simple-git deprecation notice: git.silent: logging should be configured using the `debug` library / `DEBUG` environment variable, this method will be removed."
          )
        )
      );
    };
    Git2.prototype.tags = function(options, then) {
      return this._runTask(
        tagListTask2(getTrailingOptions2(arguments)),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.rebase = function() {
      return this._runTask(
        straightThroughStringTask2(["rebase", ...getTrailingOptions2(arguments)]),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.reset = function(mode) {
      return this._runTask(
        resetTask2(getResetMode2(mode), getTrailingOptions2(arguments)),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.revert = function(commit) {
      const next = trailingFunctionArgument2(arguments);
      if (typeof commit !== "string") {
        return this._runTask(configurationErrorTask2("Commit must be a string"), next);
      }
      return this._runTask(
        straightThroughStringTask2(["revert", ...getTrailingOptions2(arguments, 0, true), commit]),
        next
      );
    };
    Git2.prototype.addTag = function(name) {
      const task = typeof name === "string" ? addTagTask2(name) : configurationErrorTask2("Git.addTag requires a tag name");
      return this._runTask(task, trailingFunctionArgument2(arguments));
    };
    Git2.prototype.addAnnotatedTag = function(tagName, tagMessage) {
      return this._runTask(
        addAnnotatedTagTask2(tagName, tagMessage),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.deleteLocalBranch = function(branchName, forceDelete, then) {
      return this._runTask(
        deleteBranchTask2(branchName, typeof forceDelete === "boolean" ? forceDelete : false),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.deleteLocalBranches = function(branchNames, forceDelete, then) {
      return this._runTask(
        deleteBranchesTask2(branchNames, typeof forceDelete === "boolean" ? forceDelete : false),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.branch = function(options, then) {
      return this._runTask(
        branchTask2(getTrailingOptions2(arguments)),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.branchLocal = function(then) {
      return this._runTask(branchLocalTask2(), trailingFunctionArgument2(arguments));
    };
    Git2.prototype.raw = function(commands) {
      const createRestCommands = !Array.isArray(commands);
      const command = [].slice.call(createRestCommands ? arguments : commands, 0);
      for (let i2 = 0; i2 < command.length && createRestCommands; i2++) {
        if (!filterPrimitives2(command[i2])) {
          command.splice(i2, command.length - i2);
          break;
        }
      }
      command.push(...getTrailingOptions2(arguments, 0, true));
      var next = trailingFunctionArgument2(arguments);
      if (!command.length) {
        return this._runTask(
          configurationErrorTask2("Raw: must supply one or more command to execute"),
          next
        );
      }
      return this._runTask(straightThroughStringTask2(command, this._trimmed), next);
    };
    Git2.prototype.submoduleAdd = function(repo, path23, then) {
      return this._runTask(addSubModuleTask2(repo, path23), trailingFunctionArgument2(arguments));
    };
    Git2.prototype.submoduleUpdate = function(args, then) {
      return this._runTask(
        updateSubModuleTask2(getTrailingOptions2(arguments, true)),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.submoduleInit = function(args, then) {
      return this._runTask(
        initSubModuleTask2(getTrailingOptions2(arguments, true)),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.subModule = function(options, then) {
      return this._runTask(
        subModuleTask2(getTrailingOptions2(arguments)),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.listRemote = function() {
      return this._runTask(
        listRemotesTask2(getTrailingOptions2(arguments)),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.addRemote = function(remoteName, remoteRepo, then) {
      return this._runTask(
        addRemoteTask2(remoteName, remoteRepo, getTrailingOptions2(arguments)),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.removeRemote = function(remoteName, then) {
      return this._runTask(removeRemoteTask2(remoteName), trailingFunctionArgument2(arguments));
    };
    Git2.prototype.getRemotes = function(verbose, then) {
      return this._runTask(getRemotesTask2(verbose === true), trailingFunctionArgument2(arguments));
    };
    Git2.prototype.remote = function(options, then) {
      return this._runTask(
        remoteTask2(getTrailingOptions2(arguments)),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.tag = function(options, then) {
      const command = getTrailingOptions2(arguments);
      if (command[0] !== "tag") {
        command.unshift("tag");
      }
      return this._runTask(straightThroughStringTask2(command), trailingFunctionArgument2(arguments));
    };
    Git2.prototype.updateServerInfo = function(then) {
      return this._runTask(
        straightThroughStringTask2(["update-server-info"]),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.pushTags = function(remote, then) {
      const task = pushTagsTask2(
        { remote: filterType2(remote, filterString2) },
        getTrailingOptions2(arguments)
      );
      return this._runTask(task, trailingFunctionArgument2(arguments));
    };
    Git2.prototype.rm = function(files) {
      return this._runTask(
        straightThroughStringTask2(["rm", "-f", ...asArray2(files)]),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.rmKeepLocal = function(files) {
      return this._runTask(
        straightThroughStringTask2(["rm", "--cached", ...asArray2(files)]),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.catFile = function(options, then) {
      return this._catFile("utf-8", arguments);
    };
    Git2.prototype.binaryCatFile = function() {
      return this._catFile("buffer", arguments);
    };
    Git2.prototype._catFile = function(format, args) {
      var handler = trailingFunctionArgument2(args);
      var command = ["cat-file"];
      var options = args[0];
      if (typeof options === "string") {
        return this._runTask(
          configurationErrorTask2("Git.catFile: options must be supplied as an array of strings"),
          handler
        );
      }
      if (Array.isArray(options)) {
        command.push.apply(command, options);
      }
      const task = format === "buffer" ? straightThroughBufferTask2(command) : straightThroughStringTask2(command);
      return this._runTask(task, handler);
    };
    Git2.prototype.diff = function(options, then) {
      const task = filterString2(options) ? configurationErrorTask2(
        "git.diff: supplying options as a single string is no longer supported, switch to an array of strings"
      ) : straightThroughStringTask2(["diff", ...getTrailingOptions2(arguments)]);
      return this._runTask(task, trailingFunctionArgument2(arguments));
    };
    Git2.prototype.diffSummary = function() {
      return this._runTask(
        diffSummaryTask2(getTrailingOptions2(arguments, 1)),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.applyPatch = function(patches) {
      const task = !filterStringOrStringArray2(patches) ? configurationErrorTask2(
        `git.applyPatch requires one or more string patches as the first argument`
      ) : applyPatchTask2(asArray2(patches), getTrailingOptions2([].slice.call(arguments, 1)));
      return this._runTask(task, trailingFunctionArgument2(arguments));
    };
    Git2.prototype.revparse = function() {
      const commands = ["rev-parse", ...getTrailingOptions2(arguments, true)];
      return this._runTask(
        straightThroughStringTask2(commands, true),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.clean = function(mode, options, then) {
      const usingCleanOptionsArray = isCleanOptionsArray2(mode);
      const cleanMode = usingCleanOptionsArray && mode.join("") || filterType2(mode, filterString2) || "";
      const customArgs = getTrailingOptions2([].slice.call(arguments, usingCleanOptionsArray ? 1 : 0));
      return this._runTask(
        cleanWithOptionsTask2(cleanMode, customArgs),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.exec = function(then) {
      const task = {
        commands: [],
        format: "utf-8",
        parser() {
          if (typeof then === "function") {
            then();
          }
        }
      };
      return this._runTask(task);
    };
    Git2.prototype.clearQueue = function() {
      return this._runTask(
        adhocExecTask2(
          () => console.warn(
            "simple-git deprecation notice: clearQueue() is deprecated and will be removed, switch to using the abortPlugin instead."
          )
        )
      );
    };
    Git2.prototype.checkIgnore = function(pathnames, then) {
      return this._runTask(
        checkIgnoreTask2(asArray2(filterType2(pathnames, filterStringOrStringArray2, []))),
        trailingFunctionArgument2(arguments)
      );
    };
    Git2.prototype.checkIsRepo = function(checkType, then) {
      return this._runTask(
        checkIsRepoTask2(filterType2(checkType, filterString2)),
        trailingFunctionArgument2(arguments)
      );
    };
    module.exports = Git2;
  }
});
init_git_error();
var GitConstructError = class extends GitError {
  constructor(config, message) {
    super(void 0, message);
    this.config = config;
  }
};
init_git_error();
init_git_error();
var GitPluginError = class extends GitError {
  constructor(task, plugin, message) {
    super(task, message);
    this.task = task;
    this.plugin = plugin;
    Object.setPrototypeOf(this, new.target.prototype);
  }
};
init_git_response_error();
init_task_configuration_error();
init_check_is_repo();
init_clean();
init_config();
init_diff_name_status();
init_grep();
init_reset();
function abortPlugin(signal) {
  if (!signal) {
    return;
  }
  const onSpawnAfter = {
    type: "spawn.after",
    action(_data, context) {
      function kill() {
        context.kill(new GitPluginError(void 0, "abort", "Abort signal received"));
      }
      signal.addEventListener("abort", kill);
      context.spawned.on("close", () => signal.removeEventListener("abort", kill));
    }
  };
  const onSpawnBefore = {
    type: "spawn.before",
    action(_data, context) {
      if (signal.aborted) {
        context.kill(new GitPluginError(void 0, "abort", "Abort already signaled"));
      }
    }
  };
  return [onSpawnBefore, onSpawnAfter];
}
function blockUnsafeOperationsPlugin(options = {}) {
  return {
    type: "spawn.args",
    action(args, { env: env2 }) {
      for (const vulnerability of ne(args, env2)) {
        if (options[vulnerability.category] !== true) {
          throw new GitPluginError(void 0, "unsafe", vulnerability.message);
        }
      }
      return args;
    }
  };
}
init_utils();
function commandConfigPrefixingPlugin(configuration) {
  const prefix = prefixedArray(configuration, "-c");
  return {
    type: "spawn.args",
    action(data) {
      return [...prefix, ...data];
    }
  };
}
init_utils();
var never = (0, import_promise_deferred2.deferred)().promise;
function completionDetectionPlugin({
  onClose = true,
  onExit = 50
} = {}) {
  function createEvents() {
    let exitCode = -1;
    const events = {
      close: (0, import_promise_deferred2.deferred)(),
      closeTimeout: (0, import_promise_deferred2.deferred)(),
      exit: (0, import_promise_deferred2.deferred)(),
      exitTimeout: (0, import_promise_deferred2.deferred)()
    };
    const result = Promise.race([
      onClose === false ? never : events.closeTimeout.promise,
      onExit === false ? never : events.exitTimeout.promise
    ]);
    configureTimeout(onClose, events.close, events.closeTimeout);
    configureTimeout(onExit, events.exit, events.exitTimeout);
    return {
      close(code) {
        exitCode = code;
        events.close.done();
      },
      exit(code) {
        exitCode = code;
        events.exit.done();
      },
      get exitCode() {
        return exitCode;
      },
      result
    };
  }
  function configureTimeout(flag, event, timeout) {
    if (flag === false) {
      return;
    }
    (flag === true ? event.promise : event.promise.then(() => delay(flag))).then(timeout.done);
  }
  return {
    type: "spawn.after",
    async action(_data, { spawned, close }) {
      const events = createEvents();
      let deferClose = true;
      let quickClose = () => void (deferClose = false);
      spawned.stdout?.on("data", quickClose);
      spawned.stderr?.on("data", quickClose);
      spawned.on("error", quickClose);
      spawned.on("close", (code) => events.close(code));
      spawned.on("exit", (code) => events.exit(code));
      try {
        await events.result;
        if (deferClose) {
          await delay(50);
        }
        close(events.exitCode);
      } catch (err) {
        close(events.exitCode, err);
      }
    }
  };
}
init_utils();
var WRONG_NUMBER_ERR = `Invalid value supplied for custom binary, requires a single string or an array containing either one or two strings`;
var WRONG_CHARS_ERR = `Invalid value supplied for custom binary, restricted characters must be removed or supply the unsafe.allowUnsafeCustomBinary option`;
function isBadArgument(arg) {
  return !arg || !/^([a-z]:)?([a-z0-9/.\\_~-]+)$/i.test(arg);
}
function toBinaryConfig(input, allowUnsafe) {
  if (input.length < 1 || input.length > 2) {
    throw new GitPluginError(void 0, "binary", WRONG_NUMBER_ERR);
  }
  const isBad = input.some(isBadArgument);
  if (isBad) {
    if (allowUnsafe) {
      console.warn(WRONG_CHARS_ERR);
    } else {
      throw new GitPluginError(void 0, "binary", WRONG_CHARS_ERR);
    }
  }
  const [binary, prefix] = input;
  return {
    binary,
    prefix
  };
}
function customBinaryPlugin(plugins, input = ["git"], allowUnsafe = false) {
  let config = toBinaryConfig(asArray(input), allowUnsafe);
  plugins.on("binary", (input2) => {
    config = toBinaryConfig(asArray(input2), allowUnsafe);
  });
  plugins.append("spawn.binary", () => {
    return config.binary;
  });
  plugins.append("spawn.args", (data) => {
    return config.prefix ? [config.prefix, ...data] : data;
  });
}
init_git_error();
function isTaskError(result) {
  return !!(result.exitCode && result.stdErr.length);
}
function getErrorMessage(result) {
  return Buffer.concat([...result.stdOut, ...result.stdErr]);
}
function errorDetectionHandler(overwrite = false, isError = isTaskError, errorMessage2 = getErrorMessage) {
  return (error, result) => {
    if (!overwrite && error || !isError(result)) {
      return error;
    }
    return errorMessage2(result);
  };
}
function errorDetectionPlugin(config) {
  return {
    type: "task.error",
    action(data, context) {
      const error = config(data.error, {
        stdErr: context.stdErr,
        stdOut: context.stdOut,
        exitCode: context.exitCode
      });
      if (Buffer.isBuffer(error)) {
        return { error: new GitError(void 0, error.toString("utf-8")) };
      }
      return {
        error
      };
    }
  };
}
init_utils();
var PluginStore = class {
  constructor() {
    this.plugins = /* @__PURE__ */ new Set();
    this.events = new EventEmitter();
  }
  on(type, listener) {
    this.events.on(type, listener);
  }
  reconfigure(type, data) {
    this.events.emit(type, data);
  }
  append(type, action) {
    const plugin = append(this.plugins, { type, action });
    return () => this.plugins.delete(plugin);
  }
  add(plugin) {
    const plugins = [];
    asArray(plugin).forEach((plugin2) => plugin2 && this.plugins.add(append(plugins, plugin2)));
    return () => {
      plugins.forEach((plugin2) => this.plugins.delete(plugin2));
    };
  }
  exec(type, data, context) {
    let output = data;
    const contextual = Object.freeze(Object.create(context));
    for (const plugin of this.plugins) {
      if (plugin.type === type) {
        output = plugin.action(output, contextual);
      }
    }
    return output;
  }
};
init_utils();
function progressMonitorPlugin(progress) {
  const progressCommand = "--progress";
  const progressMethods = ["checkout", "clone", "fetch", "pull", "push"];
  const onProgress = {
    type: "spawn.after",
    action(_data, context) {
      if (!context.commands.includes(progressCommand)) {
        return;
      }
      context.spawned.stderr?.on("data", (chunk) => {
        const message = /^([\s\S]+?):\s*(\d+)% \((\d+)\/(\d+)\)/.exec(chunk.toString("utf8"));
        if (!message) {
          return;
        }
        progress({
          method: context.method,
          stage: progressEventStage(message[1]),
          progress: asNumber(message[2]),
          processed: asNumber(message[3]),
          total: asNumber(message[4])
        });
      });
    }
  };
  const onArgs = {
    type: "spawn.args",
    action(args, context) {
      if (!progressMethods.includes(context.method)) {
        return args;
      }
      return including(args, progressCommand);
    }
  };
  return [onArgs, onProgress];
}
function progressEventStage(input) {
  return String(input.toLowerCase().split(" ", 1)) || "unknown";
}
init_utils();
function spawnOptionsPlugin(spawnOptions) {
  const options = pick(spawnOptions, ["uid", "gid"]);
  return {
    type: "spawn.options",
    action(data) {
      return { ...options, ...data };
    }
  };
}
function timeoutPlugin({
  block,
  stdErr = true,
  stdOut = true
}) {
  if (block > 0) {
    return {
      type: "spawn.after",
      action(_data, context) {
        let timeout;
        function wait() {
          timeout && clearTimeout(timeout);
          timeout = setTimeout(kill, block);
        }
        function stop() {
          context.spawned.stdout?.off("data", wait);
          context.spawned.stderr?.off("data", wait);
          context.spawned.off("exit", stop);
          context.spawned.off("close", stop);
          timeout && clearTimeout(timeout);
        }
        function kill() {
          stop();
          context.kill(new GitPluginError(void 0, "timeout", `block timeout reached`));
        }
        stdOut && context.spawned.stdout?.on("data", wait);
        stdErr && context.spawned.stderr?.on("data", wait);
        context.spawned.on("exit", stop);
        context.spawned.on("close", stop);
        wait();
      }
    };
  }
}
function suffixPathsPlugin() {
  return {
    type: "spawn.args",
    action(data) {
      const prefix = [];
      let suffix;
      function append2(args) {
        (suffix = suffix || []).push(...args);
      }
      for (let i2 = 0; i2 < data.length; i2++) {
        const param = data[i2];
        if (r(param)) {
          append2(o(param));
          continue;
        }
        if (param === "--") {
          append2(
            data.slice(i2 + 1).flatMap((item) => r(item) && o(item) || item)
          );
          break;
        }
        prefix.push(param);
      }
      return !suffix ? prefix : [...prefix, "--", ...suffix.map(String)];
    }
  };
}
init_utils();
var Git = require_git();
function gitInstanceFactory(baseDir, options) {
  const plugins = new PluginStore();
  const config = createInstanceConfig(
    baseDir && (typeof baseDir === "string" ? { baseDir } : baseDir) || {},
    options
  );
  if (!folderExists(config.baseDir)) {
    throw new GitConstructError(
      config,
      `Cannot use simple-git on a directory that does not exist`
    );
  }
  if (Array.isArray(config.config)) {
    plugins.add(commandConfigPrefixingPlugin(config.config));
  }
  plugins.add(blockUnsafeOperationsPlugin(config.unsafe));
  plugins.add(completionDetectionPlugin(config.completion));
  config.abort && plugins.add(abortPlugin(config.abort));
  config.progress && plugins.add(progressMonitorPlugin(config.progress));
  config.timeout && plugins.add(timeoutPlugin(config.timeout));
  config.spawnOptions && plugins.add(spawnOptionsPlugin(config.spawnOptions));
  plugins.add(suffixPathsPlugin());
  plugins.add(errorDetectionPlugin(errorDetectionHandler(true)));
  config.errors && plugins.add(errorDetectionPlugin(config.errors));
  customBinaryPlugin(plugins, config.binary, config.unsafe?.allowUnsafeCustomBinary);
  return new Git(config, plugins);
}
init_git_response_error();
var esm_default = gitInstanceFactory;

// electron/git-review-ops.ts
var COMMIT_CONTEXT_DIFF_MAX_CHARS = 12e4;
var COMMIT_CONTEXT_UNTRACKED_MAX = 80;
var REVIEW_FILE_CAP = 2e3;
var UNTRACKED_LINE_COUNT_CONCURRENCY = 16;
var UNTRACKED_LINE_COUNT_MAX_BYTES = 1024 * 1024;
function ghEnv(ghBin) {
  const extra = [ghBin ? path9.dirname(ghBin) : "", "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"].filter(
    (dir) => dir && dir !== "."
  );
  return { ...process.env, PATH: [...extra, process.env.PATH].filter(Boolean).join(path9.delimiter) };
}
function runGh(args, cwd, ghBin) {
  return new Promise((resolve) => {
    execFile(
      ghBin || "gh",
      args,
      { cwd, env: ghEnv(ghBin), windowsHide: true, timeout: 3e4, maxBuffer: 8 * 1024 * 1024 },
      (err, stdout) => resolve({ ok: !err, stdout: String(stdout || "") })
    );
  });
}
function gitFor(cwd, gitBin) {
  return esm_default({
    baseDir: cwd,
    binary: gitBin || "git",
    maxConcurrentProcesses: 4,
    trimmed: false,
    ...gitBin && /\s/.test(gitBin) ? { unsafe: { allowUnsafeCustomBinary: true } } : {}
  });
}
function resolveRenamePath(raw) {
  const path23 = String(raw || "").trim();
  if (!path23.includes(" => ")) {
    return path23;
  }
  const brace = path23.match(/^(.*)\{(.*) => (.*)\}(.*)$/);
  if (brace) {
    const [, prefix, , to, suffix] = brace;
    return `${prefix}${to}${suffix}`.replace(/\/{2,}/g, "/");
  }
  return path23.split(" => ").pop().trim();
}
function countsByPath(summary) {
  const map = /* @__PURE__ */ new Map();
  for (const file of summary.files) {
    map.set(resolveRenamePath(file.file), {
      added: file.binary ? 0 : file.insertions,
      removed: file.binary ? 0 : file.deletions
    });
  }
  return map;
}
async function untrackedInsertions(cwd, relPath) {
  try {
    const fullPath = path9.join(cwd, relPath);
    const stat = await fs9.stat(fullPath);
    if (!stat.isFile() || stat.size > UNTRACKED_LINE_COUNT_MAX_BYTES) {
      return 0;
    }
    const buf = await fs9.readFile(fullPath);
    if (buf.includes(0)) {
      return 0;
    }
    let lines = 0;
    for (const byte of buf) {
      if (byte === 10) {
        lines++;
      }
    }
    return buf.length > 0 && buf[buf.length - 1] !== 10 ? lines + 1 : lines;
  } catch {
    return 0;
  }
}
function capText(text, maxChars, label = "truncated") {
  const value = String(text || "");
  if (value.length <= maxChars) {
    return value;
  }
  return `${value.slice(0, maxChars)}
# ${label}: ${value.length - maxChars} chars omitted
`;
}
async function fillUntrackedCounts(cwd, files) {
  const pending = files.filter((file) => file.status === "?" && file.added === 0 && file.removed === 0);
  for (let i2 = 0; i2 < pending.length; i2 += UNTRACKED_LINE_COUNT_CONCURRENCY) {
    await Promise.all(
      pending.slice(i2, i2 + UNTRACKED_LINE_COUNT_CONCURRENCY).map(async (file) => {
        file.added = await untrackedInsertions(cwd, file.path);
      })
    );
  }
}
async function branchBase(git) {
  const candidates = [];
  try {
    const head = (await git.revparse(["--abbrev-ref", "origin/HEAD"])).trim();
    if (head) {
      candidates.push(head);
    }
  } catch {
  }
  candidates.push("origin/main", "origin/master", "main", "master");
  for (const ref of candidates) {
    try {
      const base = (await git.raw(["merge-base", "HEAD", ref])).trim();
      if (base) {
        return base;
      }
    } catch {
    }
  }
  return null;
}
async function defaultBranchName(git) {
  try {
    const head = (await git.revparse(["--abbrev-ref", "origin/HEAD"])).trim();
    if (head && head !== "origin/HEAD") {
      return head.replace(/^origin\//, "");
    }
  } catch {
  }
  for (const ref of [
    "refs/heads/main",
    "refs/heads/master",
    "refs/remotes/origin/main",
    "refs/remotes/origin/master"
  ]) {
    try {
      await git.raw(["rev-parse", "--verify", "--quiet", ref]);
      return ref.replace(/^refs\/(?:heads|remotes\/origin)\//, "");
    } catch {
    }
  }
  return null;
}
function statusLetter(file) {
  if (file.index === "?" || file.working_dir === "?") {
    return "?";
  }
  const code = file.index && file.index !== " " ? file.index : file.working_dir;
  return (code || "M").toUpperCase();
}
var isStaged = (file) => Boolean(file.index && file.index !== " " && file.index !== "?");
async function reviewList(repoPath, scope, baseRef, gitBin) {
  let cwd;
  try {
    cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review list" });
  } catch {
    return { files: [], base: null };
  }
  const git = gitFor(cwd, gitBin);
  try {
    if (scope === "branch" || scope === "lastTurn") {
      const base = scope === "branch" ? await branchBase(git) : baseRef;
      if (!base) {
        return { files: [], base: null };
      }
      const range = scope === "branch" ? `${base}...HEAD` : base;
      const summary = await git.diffSummary([range]);
      const files2 = summary.files.slice(0, REVIEW_FILE_CAP).map((file) => ({
        path: resolveRenamePath(file.file),
        added: "insertions" in file ? file.insertions : 0,
        removed: "deletions" in file ? file.deletions : 0,
        status: "M",
        staged: false
      }));
      if (scope === "lastTurn" && files2.length < REVIEW_FILE_CAP) {
        const status2 = await git.status(["--untracked-files=normal"]);
        const knownPaths = new Set(files2.map((file) => file.path));
        for (const path23 of status2.not_added) {
          if (files2.length >= REVIEW_FILE_CAP) {
            break;
          }
          if (!knownPaths.has(path23)) {
            files2.push({ path: path23, added: 0, removed: 0, status: "?", staged: false });
            knownPaths.add(path23);
          }
        }
      }
      files2.sort((a, b2) => a.path.localeCompare(b2.path));
      await fillUntrackedCounts(cwd, files2);
      return { files: files2, base };
    }
    const [status, staged, unstaged] = await Promise.all([
      // `normal` reports an untracked directory as one row instead of walking
      // every descendant. The result is also capped before per-file stat/read
      // work and before crossing the Electron IPC boundary.
      git.status(["--untracked-files=normal"]),
      git.diffSummary(["--cached"]),
      git.diffSummary([])
    ]);
    const stagedCounts = countsByPath(staged);
    const unstagedCounts = countsByPath(unstaged);
    const files = status.files.slice(0, REVIEW_FILE_CAP).map((file) => {
      const filePath = resolveRenamePath(file.path);
      const sc = stagedCounts.get(filePath) || { added: 0, removed: 0 };
      const uc = unstagedCounts.get(filePath) || { added: 0, removed: 0 };
      return {
        path: filePath,
        added: sc.added + uc.added,
        removed: sc.removed + uc.removed,
        status: statusLetter(file),
        staged: isStaged(file)
      };
    });
    files.sort((a, b2) => a.path.localeCompare(b2.path));
    await fillUntrackedCounts(cwd, files);
    return { files, base: null };
  } catch {
    return { files: [], base: null };
  }
}
async function reviewDiff(repoPath, filePath, scope, baseRef, staged, gitBin) {
  let cwd;
  try {
    cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review diff" });
  } catch {
    return "";
  }
  const git = gitFor(cwd, gitBin);
  const safe = (args) => git.diff(args).catch(() => "");
  if (scope === "branch") {
    const base = await branchBase(git);
    return base ? safe([`${base}...HEAD`, "--", filePath]) : "";
  }
  if (scope === "lastTurn") {
    return baseRef ? safe([baseRef, "--", filePath]) : "";
  }
  if (staged) {
    return safe(["--cached", "--", filePath]);
  }
  const worktree = await safe(["--", filePath]);
  if (worktree.trim()) {
    return worktree;
  }
  return new Promise((resolve) => {
    execFile(
      gitBin || "git",
      ["diff", "--no-index", "--", "/dev/null", filePath],
      { cwd, windowsHide: true, timeout: 3e4, maxBuffer: 32 * 1024 * 1024 },
      (_err, stdout) => resolve(String(stdout || ""))
    );
  });
}
async function fileDiffVsHead(repoPath, filePath, gitBin) {
  let cwd;
  try {
    cwd = resolveRequestedPathForIpc(repoPath, { purpose: "File diff" });
  } catch {
    return "";
  }
  const git = gitFor(cwd, gitBin);
  const head = await git.diff(["HEAD", "--", filePath]).catch(() => "");
  if (head.trim()) {
    return head;
  }
  const status = await git.raw(["status", "--porcelain", "--", filePath]).catch(() => "");
  if (!status.trim().startsWith("??")) {
    return "";
  }
  return new Promise((resolve) => {
    execFile(
      gitBin || "git",
      ["diff", "--no-index", "--", "/dev/null", filePath],
      { cwd, windowsHide: true, timeout: 3e4, maxBuffer: 32 * 1024 * 1024 },
      (_err, stdout) => resolve(String(stdout || ""))
    );
  });
}
async function reviewStage(repoPath, filePath, gitBin) {
  const cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review stage" });
  await gitFor(cwd, gitBin).raw(filePath ? ["add", "--", filePath] : ["add", "-A"]);
  return { ok: true };
}
async function reviewUnstage(repoPath, filePath, gitBin) {
  const cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review unstage" });
  await gitFor(cwd, gitBin).raw(filePath ? ["reset", "-q", "HEAD", "--", filePath] : ["reset", "-q", "HEAD"]);
  return { ok: true };
}
async function reviewRevert(repoPath, filePath, gitBin) {
  const cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review revert" });
  const git = gitFor(cwd, gitBin);
  if (filePath) {
    await git.raw(["checkout", "HEAD", "--", filePath]).catch(() => void 0);
    await git.raw(["clean", "-fd", "--", filePath]).catch(() => void 0);
  } else {
    await git.raw(["checkout", "HEAD", "--", "."]).catch(() => void 0);
    await git.raw(["clean", "-fd"]).catch(() => void 0);
  }
  return { ok: true };
}
async function reviewRevParse(repoPath, ref, gitBin) {
  let cwd;
  try {
    cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review rev-parse" });
  } catch {
    return null;
  }
  try {
    return (await gitFor(cwd, gitBin).revparse([ref || "HEAD"])).trim() || null;
  } catch {
    return null;
  }
}
async function reviewCommit(repoPath, message, push, gitBin) {
  const cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review commit" });
  const git = gitFor(cwd, gitBin);
  const status = await git.status();
  if (status.staged.length === 0) {
    await git.raw(["add", "-A"]);
  }
  await git.commit(message);
  if (push) {
    const fresh = await git.status();
    if (fresh.tracking) {
      await git.push();
    } else if (fresh.current) {
      await git.raw(["push", "-u", "origin", fresh.current]);
    }
  }
  return { ok: true };
}
async function reviewCommitContext(repoPath, gitBin) {
  let cwd;
  try {
    cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review commit context" });
  } catch {
    return { diff: "", recent: "" };
  }
  const git = gitFor(cwd, gitBin);
  const safe = (args) => git.diff(args).catch(() => "");
  let status;
  try {
    status = await git.status();
  } catch {
    return { diff: "", recent: "" };
  }
  let diff = capText(
    status.staged.length > 0 ? await safe(["--cached"]) : await safe(["HEAD"]),
    COMMIT_CONTEXT_DIFF_MAX_CHARS,
    "diff truncated for commit-message generation"
  );
  const untracked = status.not_added || [];
  if (untracked.length > 0) {
    const visible = untracked.slice(0, COMMIT_CONTEXT_UNTRACKED_MAX);
    const omitted = untracked.length - visible.length;
    const note = `
# New (untracked) files:
${visible.map((p2) => `#   ${p2}`).join("\n")}
` + (omitted > 0 ? `#   ... ${omitted} more omitted
` : "");
    diff = diff ? `${diff}${note}` : note;
  }
  const recent = await git.raw(["log", "-n", "10", "--pretty=format:%s"]).catch(() => "");
  return { diff: diff || "", recent: String(recent || "").trim() };
}
async function reviewPush(repoPath, gitBin) {
  const cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review push" });
  const git = gitFor(cwd, gitBin);
  const status = await git.status();
  if (status.tracking) {
    await git.push();
  } else if (status.current) {
    await git.raw(["push", "-u", "origin", status.current]);
  }
  return { ok: true };
}
async function reviewShipInfo(repoPath, ghBin) {
  let cwd;
  try {
    cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review ship info" });
  } catch {
    return { ghReady: false, pr: null };
  }
  const auth = await runGh(["auth", "status"], cwd, ghBin);
  if (!auth.ok) {
    return { ghReady: false, pr: null };
  }
  const view = await runGh(["pr", "view", "--json", "url,state,number"], cwd, ghBin);
  if (!view.ok) {
    return { ghReady: true, pr: null };
  }
  try {
    const pr = JSON.parse(view.stdout);
    return { ghReady: true, pr: pr && pr.url ? { url: pr.url, state: pr.state, number: pr.number } : null };
  } catch {
    return { ghReady: true, pr: null };
  }
}
async function reviewCreatePr(repoPath, gitBin, ghBin) {
  const cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Review create PR" });
  await reviewPush(repoPath, gitBin).catch(() => void 0);
  const created = await runGh(["pr", "create", "--fill"], cwd, ghBin);
  if (!created.ok) {
    throw new Error("gh pr create failed (is gh installed and authenticated?)");
  }
  const url = created.stdout.trim().split("\n").filter(Boolean).pop() || "";
  return { url };
}
async function repoStatus(repoPath, gitBin) {
  let cwd;
  try {
    cwd = resolveRequestedPathForIpc(repoPath, { purpose: "Repo status" });
  } catch {
    return null;
  }
  try {
    const stat = await fs9.stat(cwd);
    if (!stat.isDirectory()) {
      return null;
    }
  } catch {
    return null;
  }
  let git;
  try {
    git = gitFor(cwd, gitBin);
  } catch {
    return null;
  }
  let status;
  try {
    status = await git.status(["--untracked-files=normal"]);
  } catch {
    return null;
  }
  const detached = typeof status.detached === "boolean" ? status.detached : !status.current;
  const files = status.files.map((file) => ({
    path: file.path,
    staged: isStaged(file),
    unstaged: Boolean(file.working_dir && file.working_dir !== " " && file.working_dir !== "?"),
    untracked: file.index === "?" || file.working_dir === "?",
    conflicted: file.index === "U" || file.working_dir === "U"
  }));
  const result = {
    branch: detached ? null : status.current || null,
    defaultBranch: await defaultBranchName(git),
    detached,
    ahead: status.ahead || 0,
    behind: status.behind || 0,
    staged: files.filter((f) => f.staged).length,
    unstaged: files.filter((f) => f.unstaged).length,
    untracked: status.not_added.length,
    conflicted: status.conflicted.length,
    changed: files.length,
    added: 0,
    removed: 0,
    files: files.slice(0, 200)
  };
  try {
    const summary = await git.diffSummary(["HEAD"]);
    result.added = summary.insertions;
    result.removed = summary.deletions;
  } catch {
  }
  try {
    const untracked = status.not_added.slice(0, 500);
    for (let i2 = 0; i2 < untracked.length; i2 += UNTRACKED_LINE_COUNT_CONCURRENCY) {
      const batch = await Promise.all(
        untracked.slice(i2, i2 + UNTRACKED_LINE_COUNT_CONCURRENCY).map((path23) => untrackedInsertions(cwd, path23))
      );
      result.added += batch.reduce((sum, n) => sum + n, 0);
    }
  } catch {
  }
  return result;
}

// electron/git-root.ts
import fs10 from "node:fs";
import path10 from "node:path";
function findGitRoot(start, fsImpl = fs10) {
  let dir = start;
  for (let i2 = 0; i2 < 50; i2 += 1) {
    try {
      if (fsImpl.existsSync(path10.join(dir, ".git"))) {
        return dir;
      }
    } catch {
      return null;
    }
    const parent = path10.dirname(dir);
    if (parent === dir) {
      return null;
    }
    dir = parent;
  }
  return null;
}
async function gitRootForIpc(startPath, options = {}) {
  const fsImpl = options.fs || fs10;
  let resolved;
  try {
    resolved = resolveRequestedPathForIpc(startPath, { purpose: "Git root" });
  } catch {
    return null;
  }
  try {
    const stat = await fsImpl.promises.stat(resolved);
    const start = stat.isDirectory() ? resolved : path10.dirname(resolved);
    return findGitRoot(start, fsImpl);
  } catch {
    return findGitRoot(resolved, fsImpl);
  }
}

// electron/git-worktree-ops.ts
import { execFile as execFile2 } from "node:child_process";
import fs11 from "node:fs";
import path11 from "node:path";
function runGit(gitBin, args, cwd) {
  return new Promise((resolve, reject) => {
    execFile2(
      gitBin,
      args,
      { cwd, windowsHide: true, timeout: 3e4, maxBuffer: 8 * 1024 * 1024 },
      (err, stdout, stderr) => {
        if (err) {
          err.stderr = String(stderr || "");
          reject(err);
          return;
        }
        resolve(String(stdout || ""));
      }
    );
  });
}
function parseWorktrees(out) {
  const trees = [];
  let cur = null;
  for (const line of out.split("\n")) {
    if (line.startsWith("worktree ")) {
      if (cur) {
        trees.push(cur);
      }
      cur = { path: line.slice(9).trim(), branch: null, detached: false, bare: false, locked: false };
    } else if (!cur) {
      continue;
    } else if (line.startsWith("branch ")) {
      cur.branch = line.slice(7).trim().replace(/^refs\/heads\//, "");
    } else if (line === "detached") {
      cur.detached = true;
    } else if (line === "bare") {
      cur.bare = true;
    } else if (line.startsWith("locked")) {
      cur.locked = true;
    }
  }
  if (cur) {
    trees.push(cur);
  }
  return trees;
}
async function listWorktrees(repoPath, gitBin) {
  let resolved;
  try {
    resolved = resolveRequestedPathForIpc(repoPath, { purpose: "Worktree list" });
  } catch {
    return [];
  }
  try {
    const out = await runGit(gitBin, ["worktree", "list", "--porcelain"], resolved);
    return parseWorktrees(out).map((tree, index) => ({
      path: tree.path,
      branch: tree.branch,
      isMain: index === 0,
      detached: tree.detached,
      locked: tree.locked
    }));
  } catch {
    return [];
  }
}
function sanitizeBranch(name) {
  return String(name || "").replace(/\s+/g, "-").replace(/[^\w./-]/g, "").replace(/-{2,}/g, "-").replace(/\/{2,}/g, "/").replace(/\.{2,}/g, ".").replace(/^[-./]+|[-./]+$/g, "");
}
function slugify(name) {
  const slug = String(name || "").trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40).replace(/-+$/g, "");
  return slug || "work";
}
var TRUNK_BRANCHES = ["main", "master"];
async function gitLine(gitBin, args, cwd) {
  try {
    return (await runGit(gitBin, args, cwd)).trim();
  } catch {
    return "";
  }
}
async function defaultBranch(gitBin, cwd) {
  const remote = (await gitLine(gitBin, ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"], cwd)).replace(/^origin\//, "");
  if (remote) {
    return remote;
  }
  const configured = await gitLine(gitBin, ["config", "--get", "init.defaultBranch"], cwd);
  if (configured) {
    return configured;
  }
  for (const branch of TRUNK_BRANCHES) {
    if (await gitLine(gitBin, ["show-ref", "--verify", `refs/heads/${branch}`], cwd)) {
      return branch;
    }
  }
  return "";
}
async function ensureGitRepo(gitBin, dir) {
  let needsRoot = false;
  try {
    const inside = (await runGit(gitBin, ["rev-parse", "--is-inside-work-tree"], dir)).trim();
    if (inside !== "true") {
      await runGit(gitBin, ["init"], dir);
      needsRoot = true;
    } else {
      try {
        await runGit(gitBin, ["rev-parse", "--verify", "HEAD"], dir);
      } catch {
        needsRoot = true;
      }
    }
  } catch {
    await runGit(gitBin, ["init"], dir);
    needsRoot = true;
  }
  if (needsRoot) {
    await runGit(
      gitBin,
      [
        "-c",
        "user.email=hermes@localhost",
        "-c",
        "user.name=Hermes",
        "commit",
        "--allow-empty",
        "-m",
        "Initial commit"
      ],
      dir
    );
  }
}
async function mainRoot(gitBin, cwd) {
  const list = await listWorktrees(cwd, gitBin);
  const main = list.find((tree) => tree.isMain);
  return main ? main.path : cwd;
}
function uniqueDir(base) {
  let dir = base;
  let n = 1;
  while (fs11.existsSync(dir)) {
    n += 1;
    dir = `${base}-${n}`;
  }
  return dir;
}
async function addExistingBranchWorktree(gitBin, root, name) {
  const branch = sanitizeBranch(name);
  if (!branch) {
    throw new Error("Branch name is required.");
  }
  if (branch === await defaultBranch(gitBin, root)) {
    await runGit(gitBin, ["switch", branch], root);
    return { path: root, branch, repoRoot: root };
  }
  const dir = uniqueDir(path11.join(root, ".worktrees", slugify(branch)));
  await runGit(gitBin, ["worktree", "add", dir, branch], root);
  return { path: dir, branch, repoRoot: root };
}
async function addWorktree(repoPath, options, gitBin) {
  const resolved = resolveRequestedPathForIpc(repoPath, { purpose: "Worktree add" });
  await ensureGitRepo(gitBin, resolved);
  const root = await mainRoot(gitBin, resolved);
  const opts = options || {};
  if (opts.existingBranch) {
    return addExistingBranchWorktree(gitBin, root, opts.existingBranch);
  }
  const slug = slugify(opts.name || `work-${Date.now().toString(36)}`);
  const branch = sanitizeBranch(opts.branch) || `hermes/${slug}`;
  const dir = uniqueDir(path11.join(root, ".worktrees", slug));
  const args = ["worktree", "add", "-b", branch, dir];
  if (opts.base) {
    const base = String(opts.base);
    if (base.startsWith("origin/")) {
      const remoteBranch = base.slice("origin/".length);
      try {
        await runGit(gitBin, ["fetch", "origin", remoteBranch], root);
      } catch {
      }
      args.push("--no-track");
    }
    args.push(base);
  }
  try {
    await runGit(gitBin, args, root);
  } catch (err) {
    if (/already exists/i.test(err.stderr || "")) {
      await runGit(gitBin, ["worktree", "add", dir, branch], root);
    } else {
      throw err;
    }
  }
  return { path: dir, branch, repoRoot: root };
}
async function removeWorktree(repoPath, worktreePath, options, gitBin) {
  const resolvedRepo = resolveRequestedPathForIpc(repoPath, { purpose: "Worktree remove (repo)" });
  const resolvedTree = resolveRequestedPathForIpc(worktreePath, { purpose: "Worktree remove (tree)" });
  const root = await mainRoot(gitBin, resolvedRepo);
  const args = ["worktree", "remove"];
  if (options && options.force) {
    args.push("--force");
  }
  args.push(resolvedTree);
  await runGit(gitBin, args, root);
  return { removed: resolvedTree };
}
async function listBranches(repoPath, gitBin) {
  let resolved;
  try {
    resolved = resolveRequestedPathForIpc(repoPath, { purpose: "Branch list" });
  } catch {
    return [];
  }
  try {
    const out = await runGit(
      gitBin,
      ["for-each-ref", "--format=%(refname:short)", "--sort=-committerdate", "refs/heads"],
      resolved
    );
    const trees = await listWorktrees(resolved, gitBin);
    const pathByBranch = new Map(trees.filter((tree) => tree.branch).map((tree) => [tree.branch, tree.path]));
    const trunk = await defaultBranch(gitBin, resolved);
    return out.split("\n").map((line) => line.trim()).filter(Boolean).map((name) => ({
      name,
      checkedOut: pathByBranch.has(name),
      isDefault: Boolean(trunk && name === trunk),
      worktreePath: pathByBranch.get(name) || null
    }));
  } catch {
    return [];
  }
}
async function switchBranch(repoPath, branch, gitBin) {
  const resolved = resolveRequestedPathForIpc(repoPath, { purpose: "Branch switch" });
  const target2 = sanitizeBranch(branch);
  if (!target2) {
    throw new Error("Branch name is required.");
  }
  await runGit(gitBin, ["switch", target2], resolved);
  return { branch: target2 };
}
async function listBaseBranches(repoPath, gitBin) {
  let resolved;
  try {
    resolved = resolveRequestedPathForIpc(repoPath, { purpose: "Base branch list" });
  } catch {
    return [];
  }
  try {
    const out = await runGit(
      gitBin,
      [
        "for-each-ref",
        "--format=%(refname:short)	%(committerdate:iso)",
        "--sort=-committerdate",
        "refs/heads",
        "refs/remotes"
      ],
      resolved
    );
    const remoteDefault = await gitLine(
      gitBin,
      ["symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"],
      resolved
    );
    const localDefault = await defaultBranch(gitBin, resolved);
    return out.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => {
      const [name] = line.split("	");
      return {
        name,
        isRemote: name.startsWith("origin/"),
        // origin/HEAD when a remote exists; otherwise the local default
        // (main/master/init.defaultBranch) so a no-remote repo still flags
        // its trunk.
        isDefault: Boolean(
          remoteDefault && name === remoteDefault || !remoteDefault && localDefault && name === localDefault
        )
      };
    });
  } catch {
    return [];
  }
}

// electron/link-title-window.ts
function linkTitleWindowOptions(partitionSession) {
  return {
    show: false,
    width: 1280,
    height: 800,
    webPreferences: {
      // Deliberately throttled: this hidden window loads arbitrary user-linked
      // pages, and an unthrottled heavy page burns full CPU for the window's
      // whole lifetime. Title resolution rides load events
      // (page-title-updated / did-finish-load) plus main-process timers, none
      // of which the renderer clamp touches — hidden-page throttling only
      // slows the page's own timer-driven JS, and the grace window already
      // absorbs that.
      contextIsolation: true,
      javascript: true,
      nodeIntegration: false,
      sandbox: true,
      session: partitionSession,
      webSecurity: true
    }
  };
}
function createLinkTitleWindow(BrowserWindow3, partitionSession) {
  const window2 = new BrowserWindow3(linkTitleWindowOptions(partitionSession));
  try {
    window2.webContents.setAudioMuted(true);
  } catch {
  }
  return window2;
}
function guardLinkTitleSession(partitionSession) {
  try {
    partitionSession.on("will-download", (_event, item) => item.cancel());
  } catch {
  }
}
function readLinkTitleWindowTitle(window2) {
  try {
    if (!window2 || window2.isDestroyed()) {
      return "";
    }
    const contents = window2.webContents;
    if (!contents || contents.isDestroyed()) {
      return "";
    }
    return contents.getTitle() || "";
  } catch {
    return "";
  }
}

// electron/main-window-lifecycle.ts
function ensureMainWindow(window2, { isReady, createWindow: createWindow2, focusWindow: focusWindow2, focusExisting = true }) {
  if (!window2 || window2.isDestroyed()) {
    if (isReady) {
      createWindow2();
    }
    return;
  }
  if (focusExisting) {
    focusWindow2(window2);
  }
}

// electron/native-auth-decisions.ts
function resolveJsonBody(body) {
  return body;
}
function oauthSessionIsLive(hasNativeToken, hasCookieSession) {
  return hasNativeToken || hasCookieSession;
}
function resolveOauthRestAuth(nativeAccessToken) {
  if (nativeAccessToken) {
    return { kind: "bearer", token: nativeAccessToken };
  }
  return { kind: "cookie" };
}
function resolveReadinessProbeAuth(authMode, nativeAccessToken, connectionToken) {
  if (authMode === "oauth") {
    return resolveOauthRestAuth(nativeAccessToken);
  }
  if (authMode === "token") {
    return { kind: "token", token: connectionToken ?? null };
  }
  return { kind: "public" };
}
function oauthGuardMayHardFail(providers) {
  if (!Array.isArray(providers) || providers.length === 0) {
    return true;
  }
  const named = providers.filter((provider) => provider && typeof provider === "object" && provider.name);
  if (named.length === 0) {
    return true;
  }
  return !named.every((provider) => provider.supportsPassword);
}

// electron/native-oauth.ts
import { createHash, randomBytes } from "node:crypto";
var NATIVE_FLOW_ID = "native_pkce";
function b64url(raw) {
  return raw.toString("base64").replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function generatePkcePair(randomImpl = randomBytes) {
  const verifier = b64url(randomImpl(32));
  const challenge = b64url(createHash("sha256").update(verifier, "ascii").digest());
  return { verifier, challenge, method: "S256" };
}
function generateState(randomImpl = randomBytes) {
  return b64url(randomImpl(24));
}
function statusSupportsNativeFlow(statusBody) {
  const flows = statusBody && statusBody.auth_flows;
  return Array.isArray(flows) && flows.includes(NATIVE_FLOW_ID);
}
function resolveLoginStrategy(statusBody, opts = {}) {
  if (opts.forceEmbedded) {
    return "embedded";
  }
  return statusSupportsNativeFlow(statusBody) ? "native" : "embedded";
}
function buildNativeAuthorizeUrl(baseUrl, params) {
  const parsed = new URL(baseUrl);
  const prefix = parsed.pathname.replace(/\/+$/, "");
  const q2 = new URLSearchParams({
    code_challenge: params.challenge,
    code_challenge_method: "S256",
    redirect_uri: params.redirectUri,
    state: params.state
  });
  if (params.provider) {
    q2.set("provider", params.provider);
  }
  return `${parsed.protocol}//${parsed.host}${prefix}/auth/native/authorize?${q2.toString()}`;
}
function nativeTokenUrl(baseUrl) {
  const parsed = new URL(baseUrl);
  const prefix = parsed.pathname.replace(/\/+$/, "");
  return `${parsed.protocol}//${parsed.host}${prefix}/auth/native/token`;
}
function nativeRefreshUrl(baseUrl) {
  const parsed = new URL(baseUrl);
  const prefix = parsed.pathname.replace(/\/+$/, "");
  return `${parsed.protocol}//${parsed.host}${prefix}/auth/native/refresh`;
}
function parseLoopbackCallback(requestUrl, expectedState) {
  const parsed = new URL(requestUrl, "http://127.0.0.1");
  const error = parsed.searchParams.get("error");
  if (error) {
    const desc = parsed.searchParams.get("error_description") || "";
    throw new Error(`Gateway rejected native login: ${error}${desc ? ` (${desc})` : ""}`);
  }
  const code = parsed.searchParams.get("code") || "";
  const state = parsed.searchParams.get("state") || "";
  if (!code) {
    throw new Error("Loopback callback missing authorization code");
  }
  if (!expectedState || state !== expectedState) {
    throw new Error("Loopback callback state mismatch (possible CSRF)");
  }
  return { code };
}
function parseTokenResponse(body) {
  const accessToken = String(body?.access_token || "");
  if (!accessToken) {
    throw new Error("Gateway token response missing access_token");
  }
  const expiresAt = Number(body?.expires_at);
  return {
    accessToken,
    refreshToken: String(body?.refresh_token || ""),
    expiresAt: Number.isFinite(expiresAt) ? expiresAt : 0,
    provider: String(body?.provider || ""),
    userId: String(body?.user_id || "")
  };
}
function parseStoredTokenSet(body) {
  const accessToken = String(body?.accessToken || "");
  if (!accessToken) {
    throw new Error("Stored token set missing accessToken");
  }
  const expiresAt = Number(body?.expiresAt);
  return {
    accessToken,
    refreshToken: String(body?.refreshToken || ""),
    expiresAt: Number.isFinite(expiresAt) ? expiresAt : 0,
    provider: String(body?.provider || ""),
    userId: String(body?.userId || "")
  };
}
function tokenNeedsRefresh(tokens, nowSeconds, skewSeconds = 60) {
  if (!tokens || !Number.isFinite(tokens.expiresAt) || tokens.expiresAt <= 0) {
    return true;
  }
  return nowSeconds >= tokens.expiresAt - skewSeconds;
}

// electron/native-oauth-login.ts
import http from "node:http";
var DEFAULT_LOGIN_TIMEOUT_MS = 5 * 60 * 1e3;
var DONE_HTML = '<!doctype html><meta charset="utf-8"><title>Signed in</title><body style="font:15px system-ui;margin:3rem;text-align:center"><h2>&#10003; Signed in to Hermes</h2><p>You can close this window and return to the app.</p><script>setTimeout(()=>window.close(),800)</script>';
async function runNativeLogin(baseUrl, deps, opts = {}) {
  const createServer = deps.createServer || http.createServer;
  const timeoutMs = deps.timeoutMs ?? DEFAULT_LOGIN_TIMEOUT_MS;
  const log = deps.rememberLog || (() => void 0);
  const { verifier, challenge } = generatePkcePair();
  const state = generateState();
  return new Promise((resolve, reject) => {
    let settled = false;
    let timer = null;
    const server = createServer((req, res) => {
      const url = req.url || "/";
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end(DONE_HTML);
      if (settled) {
        return;
      }
      if (!/[?&](code|error)=/.test(url)) {
        return;
      }
      try {
        const { code } = parseLoopbackCallback(url, state);
        finishWith(async () => {
          const tokenBody = await deps.postJson(
            nativeTokenUrl(baseUrl),
            { code, code_verifier: verifier },
            { timeoutMs: 15e3 }
          );
          return parseTokenResponse(tokenBody);
        });
      } catch (error) {
        fail(error instanceof Error ? error : new Error(String(error)));
      }
    });
    const cleanup = () => {
      if (timer) {
        clearTimeout(timer);
      }
      try {
        server.close();
      } catch {
      }
    };
    const fail = (error) => {
      if (settled) {
        return;
      }
      settled = true;
      cleanup();
      reject(error);
    };
    const finishWith = (produce) => {
      if (settled) {
        return;
      }
      settled = true;
      produce().then((tokens) => {
        cleanup();
        resolve(tokens);
      }).catch((error) => {
        cleanup();
        reject(error instanceof Error ? error : new Error(String(error)));
      });
    };
    server.on("error", (err) => fail(err instanceof Error ? err : new Error(String(err))));
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address();
      if (!addr || typeof addr === "string") {
        fail(new Error("Failed to bind loopback listener for native login"));
        return;
      }
      const redirectUri = `http://127.0.0.1:${addr.port}/callback`;
      const authorizeUrl = buildNativeAuthorizeUrl(baseUrl, {
        challenge,
        redirectUri,
        state,
        provider: opts.provider
      });
      timer = setTimeout(() => {
        fail(
          new Error(
            "Native sign-in timed out. The browser window may not have completed sign-in; open Settings \u2192 Gateway and try again."
          )
        );
      }, timeoutMs);
      log(`[native-oauth] loopback listening on 127.0.0.1:${addr.port}; opening system browser`);
      deps.openExternal(authorizeUrl).catch((error) => {
        fail(
          new Error(
            `Could not open the system browser for native sign-in: ${error instanceof Error ? error.message : String(error)}`
          )
        );
      });
    });
  });
}

// electron/native-token-store.ts
function readStore(io) {
  try {
    const parsed = JSON.parse(io.readStoreText());
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}
function redactGatewayUrl(baseUrl) {
  try {
    const parsed = new URL(baseUrl);
    parsed.username = "";
    parsed.password = "";
    return `${parsed.protocol}//${parsed.host}${parsed.pathname}`;
  } catch {
    return "<invalid gateway URL>";
  }
}
function persistNativeTokenSet(baseUrl, tokens, io) {
  const store = readStore(io);
  if (tokens) {
    const secret = io.encrypt(JSON.stringify(tokens));
    if (!secret) {
      throw new Error("Secure token storage returned no encrypted payload; refusing to overwrite stored native tokens.");
    }
    store[baseUrl] = secret;
  } else {
    delete store[baseUrl];
  }
  try {
    io.writeStoreText(JSON.stringify(store));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    io.rememberLog?.(`[native-oauth] failed to persist tokens: ${detail}`);
  }
}
function loadNativeTokenSet(baseUrl, io) {
  const secret = readStore(io)[baseUrl];
  if (!secret) {
    return null;
  }
  try {
    const plaintext = io.decrypt(secret);
    if (!plaintext) {
      io.rememberLog?.(
        `[native-oauth] failed to decrypt stored tokens for ${redactGatewayUrl(baseUrl)}; keeping stored entry for retry`
      );
      return null;
    }
    return parseStoredTokenSet(JSON.parse(plaintext));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    io.rememberLog?.(`[native-oauth] failed to load stored tokens for ${redactGatewayUrl(baseUrl)}: ${detail}`);
    return null;
  }
}

// electron/oauth-net-request.ts
function serializeJsonBody(body) {
  return body === void 0 ? void 0 : Buffer.from(JSON.stringify(body));
}
function setJsonRequestHeaders(request2) {
  request2.setHeader("Content-Type", "application/json");
}

// electron/power-save.ts
function createKeepAwake(blocker, type = "prevent-app-suspension") {
  let id = null;
  const isActive = () => id !== null && blocker.isStarted(id);
  return {
    isActive,
    set(on) {
      if (on && !isActive()) {
        id = blocker.start(type);
      } else if (!on && id !== null) {
        if (blocker.isStarted(id)) {
          blocker.stop(id);
        }
        id = null;
      }
      return isActive();
    }
  };
}

// electron/primary-backend-startup.ts
var FirstRunSetupResetError = class extends Error {
  firstRunSetupReset = true;
  constructor() {
    super("First-run setup was reset before a choice completed.");
    this.name = "FirstRunSetupResetError";
  }
};
async function runPrimaryBackendStartup({
  connectRemote,
  ensureLocalRuntime,
  prepareLocalBackend,
  resolveRemote,
  waitForDecision,
  waitForLocalStart
}) {
  const savedRemote = await resolveRemote();
  if (savedRemote) {
    return { kind: "remote", connection: await connectRemote(savedRemote) };
  }
  await waitForLocalStart();
  const backend = await prepareLocalBackend();
  const decision = await waitForDecision(backend);
  if (decision === "remote-applied") {
    const appliedRemote = await resolveRemote();
    if (!appliedRemote) {
      throw new Error("First-run remote setup completed without a saved remote backend.");
    }
    return { kind: "remote", connection: await connectRemote(appliedRemote) };
  }
  if (decision === "reset") {
    throw new FirstRunSetupResetError();
  }
  return { kind: "local", backend: await ensureLocalRuntime(backend) };
}

// electron/primary-connection-rehome.ts
async function rehomePrimaryConnection({
  clearLocalBootstrapFailure,
  mode,
  notifyConnectionApplied,
  resumeFirstRunRemote,
  teardownPrimaryBackend
}) {
  let resumedFirstRunRemote = false;
  if (mode === "remote") {
    resumedFirstRunRemote = resumeFirstRunRemote();
    clearLocalBootstrapFailure();
  }
  if (resumedFirstRunRemote) {
    return { resumedFirstRunRemote: true };
  }
  await teardownPrimaryBackend({ soft: true });
  notifyConnectionApplied();
  return { resumedFirstRunRemote: false };
}

// electron/profile-delete-routing.ts
function profileNameFromDeleteRequest(request2) {
  if (!request2 || String(request2.method || "GET").toUpperCase() !== "DELETE") {
    return null;
  }
  const match = String(request2.path || "").match(/^\/api\/profiles\/([^/?#]+)(?:[?#].*)?$/);
  if (!match) {
    return null;
  }
  let raw = "";
  try {
    raw = decodeURIComponent(match[1]);
  } catch {
    return null;
  }
  const name = raw.trim();
  if (!name) {
    return null;
  }
  if (name.toLowerCase() === "default") {
    return "default";
  }
  return name.toLowerCase();
}
function decideProfileDeleteAction(profile, deps) {
  if (!profile || deps.isDefaultProfile(profile) || !deps.isValidProfileName(profile)) {
    return { action: "noop", profile: null };
  }
  if (profile === deps.primaryProfileKey()) {
    return { action: "teardown-primary", profile };
  }
  return { action: "teardown-pool", profile };
}
function resolveRouteProfile(tornDownProfile, profile) {
  return tornDownProfile ? null : profile;
}

// electron/profile-session-routing.ts
async function fetchPrimaryProfileSessions(searchParams, fetchJsonForProfile2) {
  try {
    return await fetchJsonForProfile2(null, `/api/profiles/sessions?${searchParams}`);
  } catch {
    return { sessions: [], total: 0, profile_totals: {} };
  }
}

// electron/quick-entry.ts
var DEFAULT_QUICK_ENTRY_SHORTCUT = "CommandOrControl+Shift+Space";
var QUICK_ENTRY_WINDOW_WIDTH = 640;
var QUICK_ENTRY_WINDOW_HEIGHT = 168;
var QUICK_ENTRY_TOP_FRACTION = 0.22;
var ACCELERATOR_MODIFIERS = /* @__PURE__ */ new Set([
  "alt",
  "altgr",
  "cmd",
  "cmdorctrl",
  "command",
  "commandorcontrol",
  "control",
  "ctrl",
  "meta",
  "option",
  "shift",
  "super"
]);
var ACCELERATOR_KEYS = /* @__PURE__ */ new Set([
  "backspace",
  "delete",
  "down",
  "end",
  "enter",
  "escape",
  "home",
  "insert",
  "left",
  "medianexttrack",
  "mediaplaypause",
  "mediaprevioustrack",
  "mediastop",
  "pagedown",
  "pageup",
  "plus",
  "printscreen",
  "return",
  "right",
  "space",
  "tab",
  "up",
  "volumedown",
  "volumemute",
  "volumeup"
]);
var ACCELERATOR_PUNCTUATION = /* @__PURE__ */ new Set([
  "!",
  '"',
  "#",
  "$",
  "%",
  "&",
  "'",
  "(",
  ")",
  "*",
  "+",
  ",",
  "-",
  ".",
  "/",
  ":",
  ";",
  "<",
  "=",
  ">",
  "?",
  "@",
  "[",
  "\\",
  "]",
  "^",
  "_",
  "`",
  "{",
  "|",
  "}",
  "~"
]);
function isAcceleratorKey(token) {
  if (ACCELERATOR_KEYS.has(token)) {
    return true;
  }
  if (/^f([1-9]|1[0-9]|2[0-4])$/.test(token)) {
    return true;
  }
  if (/^num(?:[0-9]|lock|dec|add|sub|mult|div)$/.test(token)) {
    return true;
  }
  return token.length === 1 && (/^[a-z0-9]$/.test(token) || ACCELERATOR_PUNCTUATION.has(token));
}
function parseQuickEntryShortcut(raw) {
  if (typeof raw !== "string" || !raw.trim()) {
    return { ok: false, reason: "empty" };
  }
  const parts = raw.split("+").map((part) => part.trim()).filter(Boolean);
  if (parts.length === 0) {
    return { ok: false, reason: "empty" };
  }
  const modifiers = [];
  let key = null;
  for (const part of parts) {
    const lower = part.toLowerCase();
    if (ACCELERATOR_MODIFIERS.has(lower)) {
      if (key) {
        return { ok: false, reason: "invalid-modifier" };
      }
      modifiers.push(lower);
      continue;
    }
    if (key) {
      return { ok: false, reason: "invalid-key" };
    }
    if (!isAcceleratorKey(lower)) {
      return { ok: false, reason: "invalid-key" };
    }
    key = lower;
  }
  if (!key) {
    return { ok: false, reason: "no-key" };
  }
  if (modifiers.length === 0) {
    return { ok: false, reason: "no-modifier" };
  }
  if (key === "escape") {
    return { ok: false, reason: "reserved" };
  }
  const seen = /* @__PURE__ */ new Set();
  const normalizedModifiers = modifiers.map((modifier) => CANONICAL_MODIFIER[modifier] ?? modifier).filter((modifier) => seen.has(modifier) ? false : (seen.add(modifier), true)).sort((left, right) => MODIFIER_ORDER.indexOf(left) - MODIFIER_ORDER.indexOf(right));
  return { accelerator: [...normalizedModifiers, canonicalKey(key)].join("+"), ok: true };
}
var CANONICAL_MODIFIER = {
  alt: "Alt",
  altgr: "AltGr",
  cmd: "Command",
  cmdorctrl: "CommandOrControl",
  command: "Command",
  commandorcontrol: "CommandOrControl",
  control: "Control",
  ctrl: "Control",
  meta: "Super",
  option: "Option",
  shift: "Shift",
  super: "Super"
};
var MODIFIER_ORDER = ["CommandOrControl", "Command", "Control", "Super", "Alt", "Option", "AltGr", "Shift"];
var CANONICAL_KEY = {
  backspace: "Backspace",
  delete: "Delete",
  down: "Down",
  end: "End",
  enter: "Enter",
  escape: "Escape",
  home: "Home",
  insert: "Insert",
  medianexttrack: "MediaNextTrack",
  mediaplaypause: "MediaPlayPause",
  mediaprevioustrack: "MediaPreviousTrack",
  mediastop: "MediaStop",
  pagedown: "PageDown",
  pageup: "PageUp",
  plus: "Plus",
  printscreen: "PrintScreen",
  return: "Return",
  right: "Right",
  space: "Space",
  tab: "Tab",
  up: "Up",
  volumedown: "VolumeDown",
  volumemute: "VolumeMute",
  volumeup: "VolumeUp",
  left: "Left"
};
function canonicalKey(key) {
  if (CANONICAL_KEY[key]) {
    return CANONICAL_KEY[key];
  }
  if (/^f([1-9]|1[0-9]|2[0-4])$/.test(key)) {
    return key.toUpperCase();
  }
  if (key.length === 1 && /^[a-z]$/.test(key)) {
    return key.toUpperCase();
  }
  return key;
}
function sanitizeQuickEntrySettings(raw) {
  const record = raw && typeof raw === "object" ? raw : {};
  const parsed = parseQuickEntryShortcut(record.shortcut);
  return {
    // Default ON: the feature is inert until the shortcut is pressed.
    enabled: record.enabled === void 0 ? true : record.enabled === true,
    shortcut: parsed.ok ? parsed.accelerator : DEFAULT_QUICK_ENTRY_SHORTCUT
  };
}
function createQuickEntryShortcut(globalShortcut2, onTrigger) {
  let active = null;
  let state = { error: null, registered: false, shortcut: DEFAULT_QUICK_ENTRY_SHORTCUT };
  const release = () => {
    if (active) {
      try {
        globalShortcut2.unregister(active);
      } catch {
      }
      active = null;
    }
  };
  return {
    apply(settings) {
      const parsed = parseQuickEntryShortcut(settings.shortcut);
      const shortcut = parsed.ok ? parsed.accelerator : settings.shortcut;
      release();
      if (!settings.enabled) {
        state = { error: null, registered: false, shortcut };
        return state;
      }
      if (!parsed.ok) {
        state = { error: "invalid", registered: false, shortcut };
        return state;
      }
      let ok = false;
      try {
        ok = globalShortcut2.isRegistered(parsed.accelerator) ? false : globalShortcut2.register(parsed.accelerator, onTrigger);
      } catch {
        ok = false;
      }
      active = ok ? parsed.accelerator : null;
      state = { error: ok ? null : "taken", registered: ok, shortcut: parsed.accelerator };
      return state;
    },
    current() {
      return state;
    },
    dispose() {
      release();
      state = { ...state, error: null, registered: false };
    }
  };
}
function quickEntryWindowBounds(workArea) {
  const width = Math.min(QUICK_ENTRY_WINDOW_WIDTH, workArea?.width ?? QUICK_ENTRY_WINDOW_WIDTH);
  const height = Math.min(QUICK_ENTRY_WINDOW_HEIGHT, workArea?.height ?? QUICK_ENTRY_WINDOW_HEIGHT);
  if (!workArea) {
    return { height, width, x: 0, y: 0 };
  }
  const x2 = Math.round(workArea.x + (workArea.width - width) / 2);
  const maxY = workArea.y + workArea.height - height;
  const y2 = Math.round(Math.min(Math.max(workArea.y, workArea.y + workArea.height * QUICK_ENTRY_TOP_FRACTION), maxY));
  return { height, width, x: x2, y: y2 };
}

// electron/quit-guard.ts
var MAX_LISTED = 4;
var NO_ACTIVE_WORK = { count: 0, titles: [] };
function normalizeActiveWork(payload) {
  if (!payload || typeof payload !== "object") {
    return NO_ACTIVE_WORK;
  }
  const raw = payload;
  const titles = Array.isArray(raw.titles) ? raw.titles.filter((title) => typeof title === "string").map((title) => title.trim()).filter(Boolean) : [];
  const count = typeof raw.count === "number" && Number.isFinite(raw.count) ? Math.max(0, Math.floor(raw.count)) : 0;
  return { count: Math.max(count, titles.length), titles };
}
function mergeActiveWork(reports) {
  const titles = [];
  let count = 0;
  for (const report of reports) {
    count = Math.max(count, report.count);
    for (const title of report.titles) {
      if (!titles.includes(title)) {
        titles.push(title);
      }
    }
  }
  return { count: Math.max(count, titles.length), titles };
}
function quitPromptFor(work, quittingForHandoff) {
  if (quittingForHandoff || work.count < 1) {
    return null;
  }
  const listed = work.titles.slice(0, MAX_LISTED);
  const remaining = work.count - listed.length;
  const lines = listed.map((title) => `\u2022 ${title}`);
  if (remaining > 0) {
    lines.push(remaining === 1 ? "\u2022 1 more" : `\u2022 ${remaining} more`);
  }
  return {
    detail: [
      lines.join("\n"),
      lines.length > 0 ? "" : null,
      "Quitting stops the agent mid-turn. Any work it has not finished writing is lost."
    ].filter((line) => line !== null).join("\n").trim(),
    message: work.count === 1 ? "Hermes is still working on 1 chat." : `Hermes is still working on ${work.count} chats.`
  };
}

// electron/tray.ts
import { app, Menu, nativeImage, Tray } from "electron";
import * as fs12 from "node:fs";
import * as path12 from "node:path";
var tray = null;
var enabled = false;
var onShow = null;
var onQuit = null;
function prefsPath() {
  return path12.join(app.getPath("userData"), "tray-prefs.json");
}
function loadTrayPrefs() {
  try {
    const raw = fs12.readFileSync(prefsPath(), "utf8");
    const parsed = JSON.parse(raw);
    enabled = parsed?.closeToTray === true;
  } catch {
    enabled = false;
  }
  return enabled;
}
function persist() {
  try {
    fs12.writeFileSync(prefsPath(), JSON.stringify({ closeToTray: enabled }));
  } catch {
  }
}
function isTrayEnabled() {
  return enabled;
}
function iconPath() {
  const candidates = [
    path12.join(process.resourcesPath ?? "", "assets", "icon.png"),
    path12.join(app.getAppPath(), "assets", "icon.png"),
    path12.join(app.getAppPath(), "..", "assets", "icon.png")
  ];
  for (const candidate of candidates) {
    if (candidate && fs12.existsSync(candidate)) {
      return candidate;
    }
  }
  return candidates[1];
}
function ensureTray() {
  if (tray) {
    return;
  }
  const image = nativeImage.createFromPath(iconPath());
  tray = new Tray(image.isEmpty() ? nativeImage.createEmpty() : image);
  tray.setToolTip("Hermes");
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: "Open Hermes", click: () => onShow?.() },
      { type: "separator" },
      { label: "Quit", click: () => onQuit?.() }
    ])
  );
  tray.on("click", () => onShow?.());
}
function setTrayEnabled(next, callbacks) {
  enabled = next;
  onShow = callbacks.onShow;
  onQuit = callbacks.onQuit;
  persist();
  if (enabled) {
    ensureTray();
  } else if (tray) {
    tray.destroy();
    tray = null;
  }
  return enabled;
}
function destroyTray() {
  if (tray) {
    tray.destroy();
    tray = null;
  }
}

// electron/remote-lifecycle.ts
import crypto2 from "node:crypto";
var LOCKFILE_SCHEMA_VERSION = 2;
var PROTOCOL_VERSION = 1;
var READY_RE = /^HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)/m;
var REMOTE_LOCK_DIR = "~/.hermes/desktop-ssh";
var SUPPORTED_REMOTE_OS = /* @__PURE__ */ new Set(["Linux", "Darwin"]);
var DEFAULT_READY_TIMEOUT_MS = 45e3;
var READY_POLL_INTERVAL_MS = 750;
function mintToken() {
  return crypto2.randomBytes(32).toString("hex");
}
function fingerprintToken(token) {
  return crypto2.createHash("sha256").update(String(token || "")).digest("hex").slice(0, 32);
}
function validateOwnershipId(ownershipId) {
  const value = String(ownershipId || "");
  if (!/^[0-9a-f]{32}$/.test(value)) {
    throw new Error("SSH ownership ID is invalid.");
  }
  return value;
}
function validateSpawnNonce(spawnNonce) {
  const value = String(spawnNonce || "");
  if (!/^[0-9a-f]{16}$/.test(value)) {
    throw new Error("SSH spawn nonce is invalid.");
  }
  return value;
}
function ownershipDirectory(ownershipId) {
  return `${REMOTE_LOCK_DIR}/${validateOwnershipId(ownershipId)}`;
}
function lockfilePath(ownershipId) {
  return `${ownershipDirectory(ownershipId)}/backend.lock.json`;
}
function spawnLogPath(ownershipId, spawnNonce) {
  return `${ownershipDirectory(ownershipId)}/${validateSpawnNonce(spawnNonce)}.log`;
}
function shq(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}
function validateRemotePath(p2) {
  const s = String(p2 || "");
  if (!s) {
    throw new Error("Remote path must not be empty.");
  }
  if (/[\x00\n\r]/.test(s)) {
    throw new Error("Unsafe remote path: contains NUL or newline.");
  }
  if (s === "~" || s.startsWith("~/") || s.startsWith("/")) {
    return;
  }
  throw new Error(`Remote path must be absolute or start with ~/: "${s}"`);
}
function expandRemotePath(p2) {
  validateRemotePath(p2);
  if (p2 === "~") {
    return '"$HOME"';
  }
  if (p2.startsWith("~/")) {
    return '"$HOME"' + shq(p2.slice(1));
  }
  return shq(p2);
}
async function locateHermes(ssh, remoteHermesPath) {
  const resolveLauncher = async (candidate) => {
    const script = `import os,shlex,sys
p=os.path.expanduser(${shq(candidate)})
out=p
try:
 data=open(p,"r",encoding="utf-8",errors="ignore").read(4096)
 for line in data.splitlines():
  words=shlex.split(line)
  if len(words)>1 and words[0]=="exec":
   target=os.path.expanduser(words[1])
   if os.path.isabs(target) and os.access(target,os.X_OK):out=target
   break
except (OSError,ValueError):pass
print(out)`;
    const resolved = (await ssh.exec(`python3 -c ${shq(script)}`)).trim();
    return resolved || candidate;
  };
  const isExecutable = async (candidate) => {
    try {
      validateRemotePath(candidate);
      const ok = (await ssh.exec(`[ -x ${expandRemotePath(candidate)} ] && echo OK || true`)).trim();
      return ok === "OK";
    } catch {
      return false;
    }
  };
  if (remoteHermesPath) {
    if (await isExecutable(remoteHermesPath)) {
      return resolveLauncher(remoteHermesPath);
    }
    const err2 = new Error(
      `The Hermes path you set is not an executable on the remote host: "${remoteHermesPath}". Check the path (it must be the full path to the \`hermes\` binary on the remote, e.g. ~/hermes-agent/.venv/bin/hermes), or clear it to auto-detect.`
    );
    err2.kind = "hermes-not-found";
    throw err2;
  }
  const candidates = [];
  try {
    const found = (await ssh.exec(`bash -lc ${shq("command -v hermes")}`)).trim();
    if (found) {
      candidates.push(found.split("\n").pop().trim());
    }
  } catch {
  }
  candidates.push("~/.local/bin/hermes");
  candidates.push("/usr/local/bin/hermes");
  candidates.push("~/.hermes/hermes-agent/venv/bin/hermes");
  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    if (await isExecutable(candidate)) {
      return resolveLauncher(candidate);
    }
  }
  const err = new Error(
    "Hermes is not installed on the remote host (could not find a `hermes` executable). Install it on the remote with:  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh  \u2014 or set the Hermes path explicitly in the SSH connection settings."
  );
  err.kind = "hermes-not-found";
  throw err;
}
async function probeHermesVersion(ssh, hermesPath) {
  try {
    const out = (await ssh.exec(`${expandRemotePath(hermesPath)} --version 2>&1`)).trim();
    return (out.split("\n")[0] || "").trim();
  } catch {
    return "";
  }
}
async function probeRemotePlatform(ssh) {
  const out = (await ssh.exec("uname -s; uname -m")).trim().split("\n");
  const osName = (out[0] || "").trim();
  const arch = (out[1] || "").trim();
  if (!SUPPORTED_REMOTE_OS.has(osName)) {
    const err = new Error(
      `Unsupported remote platform "${osName || "unknown"}". Hermes Desktop SSH mode supports Linux, macOS, and Windows remote hosts.`
    );
    err.kind = "unsupported-platform";
    throw err;
  }
  return { os: osName, arch };
}
async function probeRemoteHermesHome(ssh) {
  try {
    const out = (await ssh.exec('echo "${HERMES_HOME:-$HOME/.hermes}"')).trim().split("\n").pop();
    return out || "~/.hermes";
  } catch (cause) {
    const error = new Error("Could not resolve the remote Hermes home.");
    error.kind = "transient-transport-error";
    error.cause = cause;
    throw error;
  }
}
async function readLockfile(ssh, ownershipId) {
  const lpath = lockfilePath(ownershipId);
  let raw;
  try {
    raw = await ssh.exec(`if [ ! -e ${expandRemotePath(lpath)} ]; then exit 0; fi; cat ${expandRemotePath(lpath)}`);
  } catch (cause) {
    const error = new Error("Could not read the SSH backend ownership record.");
    error.kind = "transient-transport-error";
    error.cause = cause;
    throw error;
  }
  const text = String(raw || "").trim();
  if (!text) {
    return null;
  }
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (!parsed || parsed.schemaVersion !== LOCKFILE_SCHEMA_VERSION) {
    return null;
  }
  const pid = parsed.pid;
  const port = parsed.port;
  if (!Number.isInteger(pid) || pid <= 0 || pid > 4194304) {
    return null;
  }
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    return null;
  }
  if (parsed.ownershipId !== ownershipId || !/^[0-9a-f]{16}$/.test(parsed.spawnNonce || "")) {
    return null;
  }
  if (!/^[0-9a-f]{32}$/.test(parsed.tokenFingerprint || "")) {
    return null;
  }
  if (parsed.protocolVersion !== PROTOCOL_VERSION) {
    return null;
  }
  if (parsed.logPath !== spawnLogPath(ownershipId, parsed.spawnNonce)) {
    return null;
  }
  for (const field of ["profile", "hermesPath", "hermesHome", "logPath", "startedAt"]) {
    if (typeof parsed[field] !== "string" || parsed[field].length > 1024) {
      return null;
    }
  }
  return parsed;
}
async function writeLockfile(ssh, ownershipId, lock) {
  const directory = ownershipDirectory(ownershipId);
  const lpath = lockfilePath(ownershipId);
  const temporaryPath = `${directory}/.${crypto2.randomBytes(8).toString("hex")}.lock.tmp`;
  const json = JSON.stringify({ ...lock, schemaVersion: LOCKFILE_SCHEMA_VERSION });
  await ssh.exec(
    `umask 077 && mkdir -p ${expandRemotePath(directory)} && printf '%s' ${shq(json)} > ${expandRemotePath(temporaryPath)} && mv -f ${expandRemotePath(temporaryPath)} ${expandRemotePath(lpath)}`
  );
}
async function removeLockfile(ssh, ownershipId) {
  const lpath = lockfilePath(ownershipId);
  try {
    await ssh.exec(`rm -f ${expandRemotePath(lpath)}`);
  } catch {
  }
}
async function remotePidAlive(ssh, pid) {
  if (!pid || !Number.isInteger(Number(pid))) {
    return false;
  }
  try {
    const out = (await ssh.exec(`kill -0 ${Number(pid)} 2>/dev/null && echo ALIVE || echo DEAD`)).trim();
    return out === "ALIVE";
  } catch (cause) {
    const error = new Error("Could not verify the SSH backend process.");
    error.kind = "transient-transport-error";
    error.cause = cause;
    throw error;
  }
}
async function pidIsOurDashboard(ssh, pid, spawnNonce, hermesPath = "") {
  if (!pid || !/^[0-9a-f]{16}$/.test(String(spawnNonce || "")) || !hermesPath) {
    return false;
  }
  try {
    const script = `import os,shlex,subprocess,sys
pid=${Number(pid)}
expected=os.path.expanduser(${shq(hermesPath)})
nonce=${shq(spawnNonce)}
try:
 raw=open(f"/proc/{pid}/cmdline","rb").read()
 args=[x.decode("utf-8","surrogateescape") for x in raw.split(b"\\0") if x]
except OSError:
 line=subprocess.check_output(["ps","-o","command=","-p",str(pid)],text=True).strip()
 args=shlex.split(line)
ok=False
try:
 serve=args.index("serve")
 owner=args.index("--ssh-owner-nonce",serve+1)
 direct=args[0]==expected
 python_entry=len(args)>1 and args[1]==expected and os.path.basename(args[0]).startswith("python")
 ok=(direct or python_entry) and "--isolated" in args[serve+1:] and args[owner+1]==nonce
except (ValueError,IndexError):pass
print("OWNED" if ok else "FOREIGN")`;
    const out = await ssh.exec(`python3 -c ${shq(script)}`);
    return String(out || "").trim() === "OWNED";
  } catch (cause) {
    const error = new Error("Could not verify SSH backend process ownership.");
    error.kind = "transient-transport-error";
    error.cause = cause;
    throw error;
  }
}
async function cleanupStale(ssh, ownershipId, lock, pidAlive = true) {
  if (pidAlive && lock && await pidIsOurDashboard(ssh, lock.pid, lock.spawnNonce, lock.hermesPath)) {
    try {
      const result = (await ssh.exec(
        `kill ${Number(lock.pid)} && i=0; while kill -0 ${Number(lock.pid)} 2>/dev/null; do i=$((i+1)); [ "$i" -ge 50 ] && exit 1; sleep 0.1; done`
      )).trim();
      void result;
    } catch (cause) {
      const error = new Error("Could not terminate the stale SSH backend.");
      error.kind = "transient-transport-error";
      error.cause = cause;
      throw error;
    }
  }
  const expectedLogPath = lock?.spawnNonce ? spawnLogPath(ownershipId, lock.spawnNonce) : "";
  if (lock?.logPath === expectedLogPath) {
    try {
      await ssh.exec(`rm -f ${expandRemotePath(lock.logPath)}`);
    } catch {
    }
  }
  await removeLockfile(ssh, ownershipId);
}
function buildSpawnCommand(hermesPath, profile, opts = {}) {
  const hermes = expandRemotePath(hermesPath);
  const profileArgs = profile ? `--profile ${shq(profile)} ` : "";
  const logPath = expandRemotePath(opts.logPath);
  const tokenFilePath = opts.tokenFilePath;
  const tokenArg = tokenFilePath ? ` --ssh-session-token-file ${expandRemotePath(tokenFilePath)}` : "";
  const ownerArg = opts.spawnNonce ? ` --ssh-owner-nonce ${validateSpawnNonce(opts.spawnNonce)}` : "";
  const subCmd = `serve --isolated --host 127.0.0.1 --port 0${tokenArg}${ownerArg}`;
  const dashCmd = `env HERMES_DESKTOP=1 ${hermes} ${profileArgs}${subCmd}`;
  return `mkdir -p "$(dirname ${logPath})" && "$(command -v setsid || echo nohup)" sh -c ${shq(`${dashCmd} </dev/null >> ${logPath} 2>&1 & echo $!`)}`;
}
async function remoteSupportsSshOwnership(ssh, hermesPath) {
  const hermes = expandRemotePath(hermesPath);
  const out = await ssh.exec(
    `help="$(${hermes} serve --help 2>&1)"; printf '%s' "$help" | grep -q ssh-session-token-file && printf '%s' "$help" | grep -q ssh-owner-nonce && echo YES || echo NO`
  );
  return String(out || "").trim().endsWith("YES");
}
async function scrapeReadyPort(ssh, logPath, { timeoutMs = DEFAULT_READY_TIMEOUT_MS, isAlive, signal } = {}) {
  const deadline = Date.now() + timeoutMs;
  const remoteLog = expandRemotePath(logPath);
  while (Date.now() < deadline) {
    assertNotAborted(signal);
    if (isAlive && !await isAlive()) {
      const err2 = new Error("Remote dashboard process exited before announcing its port.");
      err2.kind = "spawn-failed";
      throw err2;
    }
    let tail;
    try {
      tail = await ssh.exec(`cat ${remoteLog} 2>/dev/null || true`);
    } catch {
      tail = "";
    }
    const m = READY_RE.exec(String(tail || ""));
    if (m) {
      return parseInt(m[1], 10);
    }
    await new Promise((r2) => setTimeout(r2, READY_POLL_INTERVAL_MS));
  }
  const err = new Error(`Timed out waiting for the remote dashboard to announce its port (${timeoutMs}ms).`);
  err.kind = "ready-timeout";
  throw err;
}
async function spawnRemoteDashboard(ssh, { hermesPath, profile, token, ownershipId }) {
  if (!await remoteSupportsSshOwnership(ssh, hermesPath)) {
    const err = new Error(
      "The remote Hermes install does not support --ssh-session-token-file and --ssh-owner-nonce. Update Hermes on the remote host to continue using Desktop SSH mode."
    );
    err.kind = "update-required";
    throw err;
  }
  const spawnNonce = crypto2.randomBytes(8).toString("hex");
  const tokenDir = ownershipDirectory(ownershipId);
  const tokenFilePath = `${tokenDir}/${spawnNonce}.token`;
  const logPath = spawnLogPath(ownershipId, spawnNonce);
  const tokenUploadPy = `import os,sys,stat
p=os.path.expanduser(${shq(tokenFilePath)})
d=os.path.dirname(p)
n=os.path.basename(p)
os.makedirs(d,mode=0o700,exist_ok=True)
df=os.O_RDONLY|getattr(os,"O_DIRECTORY",0)|getattr(os,"O_NOFOLLOW",0)
dd=os.open(d,df)
try:
 s=os.fstat(dd)
 if not stat.S_ISDIR(s.st_mode):raise SystemExit("unsafe token directory")
 if hasattr(os,"getuid") and s.st_uid!=os.getuid():raise SystemExit("token directory owner mismatch")
 if (s.st_mode&0o777)!=0o700:os.fchmod(dd,0o700)
 fl=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0)
 now=__import__("time").time()
 for stale in os.listdir(dd):
  if stale.endswith(".token") and len(stale)==22:
   try:
    ss=os.stat(stale,dir_fd=dd,follow_symlinks=False)
    if stat.S_ISREG(ss.st_mode) and now-ss.st_mtime>3600:os.unlink(stale,dir_fd=dd)
   except OSError:pass
 fd=os.open(n,fl,0o600,dir_fd=dd)
 try:os.write(fd,sys.stdin.buffer.read())
 except BaseException:
  try:os.unlink(n,dir_fd=dd)
  except OSError:pass
  raise
 finally:os.close(fd)
finally:os.close(dd)`;
  try {
    await ssh.exec(`python3 -c ${shq(tokenUploadPy)}`, { stdinData: token });
  } catch (error) {
    try {
      await ssh.exec(`rm -f ${expandRemotePath(tokenFilePath)}`);
    } catch {
    }
    throw error;
  }
  let out;
  try {
    out = await ssh.exec(buildSpawnCommand(hermesPath, profile, { spawnNonce, tokenFilePath, logPath }));
  } catch (error) {
    try {
      await ssh.exec(`rm -f ${expandRemotePath(tokenFilePath)}`);
    } catch {
    }
    throw error;
  }
  const pid = parseInt(
    String(out || "").trim().split("\n").pop(),
    10
  );
  if (!Number.isInteger(pid) || pid <= 0) {
    try {
      await ssh.exec(`rm -f ${expandRemotePath(tokenFilePath)}`);
    } catch {
    }
    const err = new Error("Failed to launch the remote dashboard (no pid returned).");
    err.kind = "spawn-failed";
    throw err;
  }
  return { pid, spawnNonce, logPath, tokenFilePath };
}
async function cancelForwardSafe(deps, localPort, remotePort) {
  if (typeof deps.cancelForward !== "function") {
    return;
  }
  try {
    await deps.cancelForward(localPort, remotePort);
  } catch {
  }
}
function assertNotAborted(signal) {
  if (signal?.aborted) {
    const error = new Error("SSH bootstrap was cancelled.");
    error.kind = "superseded";
    throw error;
  }
}
function isForwardBindCollision(error) {
  return /address already in use|cannot listen to port|bind.*failed/i.test(String(error?.message || error || ""));
}
async function openForward(deps, remotePort, attempts = 3) {
  let lastError;
  for (let attempt = 0; attempt < attempts; attempt++) {
    const localPort = await deps.pickLocalPort();
    try {
      await deps.forward(localPort, remotePort);
      return localPort;
    } catch (error) {
      lastError = error;
      if (!isForwardBindCollision(error) || attempt === attempts - 1) {
        throw error;
      }
    }
  }
  throw lastError;
}
async function adoptOwnedServedToken(adoptServedToken, baseUrl, expectedToken, ssh, pid, label) {
  const token = await adoptServedToken(baseUrl, expectedToken, {
    childAlive: () => true,
    label
  });
  if (!await remotePidAlive(ssh, pid)) {
    const error = new Error(`${label} exited while its served token was being resolved.`);
    error.kind = token === expectedToken ? "spawn-failed" : "foreign-backend";
    throw error;
  }
  return token;
}
async function connect(deps) {
  const {
    ssh,
    profile = "",
    remoteHermesPath = "",
    ownershipId,
    forward,
    pickLocalPort: pickLocalPort2,
    waitForHermes: waitForHermes2,
    probeReuseProof,
    adoptServedToken,
    rememberLog: rememberLog2 = () => {
    },
    readyTimeoutMs = DEFAULT_READY_TIMEOUT_MS,
    signal
  } = deps;
  const log = (msg) => rememberLog2(`[ssh-lifecycle] ${msg}`);
  assertNotAborted(signal);
  const platform = await probeRemotePlatform(ssh);
  log(`remote platform ${platform.os}/${platform.arch}`);
  const hermesPath = await locateHermes(ssh, remoteHermesPath);
  log(`located hermes at ${hermesPath}`);
  const hermesVersion = await probeHermesVersion(ssh, hermesPath);
  if (hermesVersion) {
    log(`remote hermes version: ${hermesVersion}`);
  }
  const reuseToken = deps.reuseToken || "";
  const hermesHome = await probeRemoteHermesHome(ssh);
  const lock = await readLockfile(ssh, ownershipId);
  if (lock) {
    const pidAlive = await remotePidAlive(ssh, lock.pid);
    const owned = pidAlive && await pidIsOurDashboard(ssh, lock.pid, lock.spawnNonce, lock.hermesPath);
    const reusable = pidAlive && owned && lock.port > 0 && lock.profile === profile && Boolean(reuseToken) && lock.tokenFingerprint === fingerprintToken(reuseToken) && lock.hermesPath === hermesPath && lock.hermesHome === hermesHome;
    if (reusable) {
      assertNotAborted(signal);
      const localPort2 = await openForward(deps, lock.port);
      try {
        const baseUrl = `http://127.0.0.1:${localPort2}`;
        let reuseClassification;
        try {
          reuseClassification = await probeReuseProof(baseUrl, reuseToken, lock.spawnNonce);
        } catch (cause) {
          const error = new Error("Could not verify the existing SSH backend.");
          error.kind = "transient-transport-error";
          error.cause = cause;
          throw error;
        }
        if (reuseClassification === "authenticated-stale") {
          assertNotAborted(signal);
          await cancelForwardSafe(deps, localPort2, lock.port);
          await cleanupStale(ssh, ownershipId, lock);
        } else if (reuseClassification === "authenticated-ok") {
          const token = await adoptOwnedServedToken(
            adoptServedToken,
            baseUrl,
            reuseToken,
            ssh,
            lock.pid,
            "reused remote dashboard"
          );
          assertNotAborted(signal);
          log(`reusing remote dashboard pid=${lock.pid} port=${lock.port}`);
          return {
            baseUrl,
            token,
            tokenFingerprint: fingerprintToken(token),
            remotePort: lock.port,
            localPort: localPort2,
            pid: lock.pid,
            reused: true,
            platform,
            hermesPath,
            hermesVersion,
            ownershipId,
            spawnNonce: lock.spawnNonce,
            logPath: lock.logPath
          };
        } else {
          const error = new Error("SSH reuse proof returned an invalid classification.");
          error.kind = "transient-transport-error";
          throw error;
        }
      } catch (error) {
        await cancelForwardSafe(deps, localPort2, lock.port);
        throw error;
      }
    } else {
      assertNotAborted(signal);
      await cleanupStale(ssh, ownershipId, lock, pidAlive);
    }
  }
  assertNotAborted(signal);
  const spawnToken = mintToken();
  const { pid, spawnNonce, logPath, tokenFilePath } = await spawnRemoteDashboard(ssh, {
    hermesPath,
    profile,
    token: spawnToken,
    ownershipId
  });
  log(`spawned remote dashboard pid=${pid}`);
  const ownedSpawn = {
    ownershipId,
    spawnNonce,
    pid,
    port: 0,
    profile,
    hermesPath,
    hermesHome,
    logPath,
    tokenFingerprint: fingerprintToken(spawnToken),
    protocolVersion: PROTOCOL_VERSION,
    startedAt: (/* @__PURE__ */ new Date()).toISOString()
  };
  let localPort = 0;
  let remotePort = 0;
  try {
    await writeLockfile(ssh, ownershipId, ownedSpawn);
    remotePort = await scrapeReadyPort(ssh, logPath, {
      timeoutMs: readyTimeoutMs,
      isAlive: () => remotePidAlive(ssh, pid),
      signal
    });
    assertNotAborted(signal);
    log(`remote dashboard bound port ${remotePort}`);
    localPort = await openForward(deps, remotePort);
    assertNotAborted(signal);
    const baseUrl = `http://127.0.0.1:${localPort}`;
    await waitForHermes2(baseUrl, spawnToken);
    assertNotAborted(signal);
    const token = await adoptOwnedServedToken(adoptServedToken, baseUrl, spawnToken, ssh, pid, "remote dashboard");
    assertNotAborted(signal);
    const tokenFingerprint = fingerprintToken(token);
    await writeLockfile(ssh, ownershipId, { ...ownedSpawn, port: remotePort, tokenFingerprint });
    assertNotAborted(signal);
    return {
      baseUrl,
      token,
      tokenFingerprint,
      remotePort,
      localPort,
      pid,
      reused: false,
      platform,
      hermesPath,
      hermesVersion,
      ownershipId,
      spawnNonce,
      logPath
    };
  } catch (error) {
    if (localPort && remotePort) {
      await cancelForwardSafe(deps, localPort, remotePort);
    }
    try {
      await ssh.exec(`rm -f ${expandRemotePath(tokenFilePath)}`);
    } catch {
    }
    await cleanupStale(ssh, ownershipId, ownedSpawn);
    throw error;
  }
}

// electron/remote-liveness.ts
var REMOTE_LIVENESS_TIMEOUT_MS = 1e4;
var REMOTE_LIVENESS_FAILURE_LIMIT = 3;
var REMOTE_LIVENESS_FAILURE_WINDOW_MS = 6e4;
var RemoteRevalidationCoordinator = class {
  #inflightByConnection = /* @__PURE__ */ new WeakMap();
  run(connection, task) {
    const existing = this.#inflightByConnection.get(connection);
    if (existing) {
      return existing;
    }
    const pending = Promise.resolve().then(task);
    const clear = () => {
      if (this.#inflightByConnection.get(connection) === pending) {
        this.#inflightByConnection.delete(connection);
      }
    };
    this.#inflightByConnection.set(connection, pending);
    void pending.then(clear, clear);
    return pending;
  }
};
var RemoteLivenessTracker = class {
  #failureLimit;
  #failureWindowMs;
  #failuresByBaseUrl = /* @__PURE__ */ new Map();
  #now;
  constructor(failureLimit = REMOTE_LIVENESS_FAILURE_LIMIT, failureWindowMs = REMOTE_LIVENESS_FAILURE_WINDOW_MS, now = Date.now) {
    if (!Number.isInteger(failureLimit) || failureLimit < 1) {
      throw new Error("Remote liveness failure limit must be a positive integer.");
    }
    if (!Number.isFinite(failureWindowMs) || failureWindowMs < 1) {
      throw new Error("Remote liveness failure window must be positive.");
    }
    this.#failureLimit = failureLimit;
    this.#failureWindowMs = failureWindowMs;
    this.#now = now;
  }
  recordSuccess(baseUrl) {
    this.#failuresByBaseUrl.delete(baseUrl);
  }
  recordFailure(baseUrl) {
    const now = this.#now();
    const previous = this.#failuresByBaseUrl.get(baseUrl);
    const withinFailureWindow = previous && now - previous.lastFailureAt <= this.#failureWindowMs;
    const failures = (withinFailureWindow ? previous.failures : 0) + 1;
    const shouldReset = failures >= this.#failureLimit;
    if (shouldReset) {
      this.#failuresByBaseUrl.delete(baseUrl);
    } else {
      this.#failuresByBaseUrl.set(baseUrl, { failures, lastFailureAt: now });
    }
    return { failures, shouldReset };
  }
  clear() {
    this.#failuresByBaseUrl.clear();
  }
};
async function revalidatePooledRemoteBackends({
  entries,
  log,
  probe,
  stopBackend,
  tracker
}) {
  const remotes = [...entries].filter(([, entry]) => !entry.process && entry.remoteBaseUrl);
  const dropped = [];
  await Promise.all(
    remotes.map(async ([profile, entry]) => {
      const baseUrl = String(entry.remoteBaseUrl).replace(/\/+$/, "");
      try {
        await probe(`${baseUrl}/api/status`, { timeoutMs: REMOTE_LIVENESS_TIMEOUT_MS });
        tracker.recordSuccess(baseUrl);
      } catch {
        const failure = tracker.recordFailure(baseUrl);
        if (!failure.shouldReset) {
          log(
            `Pooled remote backend for profile "${profile}" failed liveness probe (${failure.failures}/${REMOTE_LIVENESS_FAILURE_LIMIT}); keeping descriptor for retry.`
          );
          return;
        }
        log(`Pooled remote backend for profile "${profile}" failed liveness probe; dropping stale descriptor.`);
        stopBackend(profile);
        dropped.push(profile);
      }
    })
  );
  return { dropped };
}
async function revalidateRemoteConnection({
  connectionPromise,
  currentConnectionPromise,
  log,
  probe,
  resetConnection,
  tracker
}) {
  let connection;
  try {
    connection = await connectionPromise;
  } catch {
    return { ok: true, rebuilt: false };
  }
  if (currentConnectionPromise() !== connectionPromise) {
    return { ok: true, rebuilt: false };
  }
  if (connection.mode !== "remote" || !connection.baseUrl) {
    return { ok: true, rebuilt: false };
  }
  const baseUrl = connection.baseUrl.replace(/\/+$/, "");
  try {
    await probe(`${baseUrl}/api/status`, { timeoutMs: REMOTE_LIVENESS_TIMEOUT_MS });
    if (currentConnectionPromise() !== connectionPromise) {
      return { ok: true, rebuilt: false };
    }
    tracker.recordSuccess(baseUrl);
    return { ok: true, rebuilt: false };
  } catch {
    if (currentConnectionPromise() !== connectionPromise) {
      return { ok: true, rebuilt: false };
    }
    const failure = tracker.recordFailure(baseUrl);
    if (!failure.shouldReset) {
      log(
        `Cached remote Hermes backend failed liveness probe (${failure.failures}/${REMOTE_LIVENESS_FAILURE_LIMIT}); keeping connection for retry.`
      );
      return { ok: true, rebuilt: false };
    }
    log("Cached remote Hermes backend failed liveness probe; dropping stale connection.");
    resetConnection();
    return { ok: true, rebuilt: true };
  }
}

// electron/session-windows.ts
import { pathToFileURL } from "node:url";
var SESSION_WINDOW_MIN_WIDTH = 420;
var SESSION_WINDOW_MIN_HEIGHT = 620;
function chatWindowWebPreferences(preloadPath) {
  return {
    preload: preloadPath,
    contextIsolation: true,
    webviewTag: true,
    sandbox: true,
    nodeIntegration: false,
    devTools: true,
    autoplayPolicy: "no-user-gesture-required"
  };
}
function buildSessionWindowUrl(sessionId, { devServer, rendererIndexPath, watch } = {}) {
  const query = `?win=secondary${watch ? "&watch=1" : ""}`;
  const route = `#/${encodeURIComponent(sessionId)}`;
  if (devServer) {
    const base = devServer.endsWith("/") ? devServer.slice(0, -1) : devServer;
    return `${base}/${query}${route}`;
  }
  return `${pathToFileURL(rendererIndexPath).toString()}${query}${route}`;
}
var INSTANCE_CASCADE_OFFSET = 32;
function instanceWindowBounds(base, fallback) {
  if (!base) {
    return fallback;
  }
  return {
    width: base.width,
    height: base.height,
    x: base.x + INSTANCE_CASCADE_OFFSET,
    y: base.y + INSTANCE_CASCADE_OFFSET
  };
}
function createSessionWindowRegistry() {
  const windows = /* @__PURE__ */ new Map();
  function openOrFocus(sessionId, factory) {
    const key = typeof sessionId === "string" ? sessionId.trim() : "";
    if (!key) {
      return null;
    }
    const existing = windows.get(key);
    if (existing && !existing.isDestroyed()) {
      if (typeof existing.isMinimized === "function" && existing.isMinimized()) {
        existing.restore?.();
      }
      if (typeof existing.isVisible === "function" && !existing.isVisible()) {
        existing.show?.();
      }
      existing.focus?.();
      return existing;
    }
    const win = factory(key);
    if (!win) {
      return null;
    }
    windows.set(key, win);
    win.on?.("closed", () => {
      if (windows.get(key) === win) {
        windows.delete(key);
      }
    });
    return win;
  }
  return {
    openOrFocus,
    get: (key) => windows.get(key),
    has: (key) => windows.has(key),
    get size() {
      return windows.size;
    }
  };
}

// electron/spawn-helper-perms.ts
import {
  chmodSync as realChmodSync,
  existsSync as realExistsSync,
  readdirSync as realReaddirSync,
  statSync as realStatSync
} from "node:fs";
import { join as join2 } from "node:path";
var EXEC_BITS = 73;
function writableNodePtyRoot(nodePtyRoot) {
  return nodePtyRoot.replace(/app\.asar(?!\.unpacked)/, "app.asar.unpacked");
}
var defaultFs = {
  existsSync: realExistsSync,
  readdirSync: (path23) => realReaddirSync(path23),
  statSync: (path23) => realStatSync(path23),
  chmodSync: realChmodSync
};
function needsExecBit(mode) {
  return (mode & EXEC_BITS) !== EXEC_BITS;
}
function withExecBits(mode) {
  return mode | EXEC_BITS;
}
function spawnHelperCandidates(nodePtyRoot, fs20 = defaultFs) {
  const candidates = [];
  const prebuilds = join2(nodePtyRoot, "prebuilds");
  if (fs20.existsSync(prebuilds)) {
    for (const entry of fs20.readdirSync(prebuilds)) {
      candidates.push(join2(prebuilds, entry, "spawn-helper"));
    }
  }
  candidates.push(join2(nodePtyRoot, "build", "Release", "spawn-helper"));
  return candidates;
}
function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}
function ensureSpawnHelperExecutable(nodePtyRoot, fs20 = defaultFs) {
  const result = { fixed: [], errors: [] };
  const writableRoot = writableNodePtyRoot(nodePtyRoot);
  for (const path23 of spawnHelperCandidates(writableRoot, fs20)) {
    if (!fs20.existsSync(path23)) {
      continue;
    }
    let mode;
    try {
      mode = fs20.statSync(path23).mode;
    } catch (error) {
      result.errors.push({ path: path23, error: errorMessage(error) });
      continue;
    }
    if (!needsExecBit(mode)) {
      continue;
    }
    try {
      fs20.chmodSync(path23, withExecBits(mode));
      result.fixed.push(path23);
    } catch (error) {
      result.errors.push({ path: path23, error: errorMessage(error) });
    }
  }
  return result;
}

// electron/ssh-bootstrap-coordinator.ts
import crypto3 from "node:crypto";
function sshConfigFingerprint(scope, config) {
  const parts = [
    scope,
    config.host,
    config.user,
    config.port,
    config.keyPath,
    config.remoteHermesPath,
    config.remoteProfile,
    config.effectiveConfigFingerprint
  ];
  return crypto3.createHash("sha256").update(JSON.stringify(parts.map((value) => value ?? ""))).digest("hex");
}
function createBootstrapCoordinator() {
  const active = /* @__PURE__ */ new Set();
  const pending = /* @__PURE__ */ new Map();
  const generations = /* @__PURE__ */ new Map();
  const drains = /* @__PURE__ */ new Map();
  function start(scope, fingerprint, run) {
    const current = pending.get(scope);
    if (current?.fingerprint === fingerprint) {
      return current.promise;
    }
    current?.controller.abort();
    const generation = (generations.get(scope) || 0) + 1;
    generations.set(scope, generation);
    const controller = new AbortController();
    const forceCleanups = /* @__PURE__ */ new Set();
    const lease = {
      signal: controller.signal,
      onForceCleanup(cleanup) {
        forceCleanups.add(cleanup);
        return () => forceCleanups.delete(cleanup);
      },
      isCurrent: () => !controller.signal.aborted && generations.get(scope) === generation,
      assertCurrent() {
        if (!this.isCurrent()) {
          const error = new Error("SSH bootstrap was superseded by newer connection settings.");
          error.kind = "superseded";
          throw error;
        }
      }
    };
    const drain = drains.get(scope) || Promise.resolve();
    const predecessor = current ? Promise.allSettled([current.promise, drain]) : drain;
    const entry = { controller, fingerprint, forceCleanups, generation, promise: null, scope };
    const promise = predecessor.then(() => {
      lease.assertCurrent();
      return run(lease);
    }).finally(() => {
      forceCleanups.clear();
      active.delete(entry);
      if (pending.get(scope)?.generation === generation) {
        pending.delete(scope);
      }
    });
    entry.promise = promise;
    active.add(entry);
    pending.set(scope, entry);
    return promise;
  }
  function cancel(scope) {
    pending.get(scope)?.controller.abort();
  }
  async function cancelAndWait(scope) {
    let release;
    const barrier = new Promise((resolve) => {
      release = resolve;
    });
    drains.set(scope, barrier);
    const entries = [...active].filter((entry) => entry.scope === scope);
    for (const entry of entries) {
      entry.controller.abort();
    }
    try {
      await Promise.allSettled(entries.map((entry) => entry.promise));
    } finally {
      if (drains.get(scope) === barrier) {
        drains.delete(scope);
      }
      release();
    }
  }
  function cancelAll() {
    for (const entry of active) {
      entry.controller.abort();
    }
  }
  async function forceCleanupAll() {
    const cleanups = [...active].flatMap((entry) => [...entry.forceCleanups]);
    await Promise.allSettled(cleanups.map((cleanup) => cleanup()));
  }
  function promises() {
    return [...active].map((entry) => entry.promise);
  }
  return { active, cancel, cancelAll, cancelAndWait, forceCleanupAll, pending, promises, start };
}

// electron/ssh-config.ts
import fs13 from "node:fs";
import os4 from "node:os";
import path13 from "node:path";
function parseSshConfigHosts(text) {
  const hosts = [];
  const seen = /* @__PURE__ */ new Set();
  for (const rawLine of String(text || "").split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const m = /^host\s+(.+)$/i.exec(line);
    if (!m) {
      continue;
    }
    for (const pattern of m[1].split(/\s+/)) {
      if (!pattern || pattern.includes("*") || pattern.includes("?") || pattern.startsWith("!")) {
        continue;
      }
      if (!seen.has(pattern)) {
        seen.add(pattern);
        hosts.push(pattern);
      }
    }
  }
  return hosts;
}
function parseSshConfigIncludes(text) {
  const includes = [];
  for (const rawLine of String(text || "").split("\n")) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }
    const m = /^include\s+(.+)$/i.exec(line);
    if (!m) {
      continue;
    }
    for (const token of m[1].split(/\s+/)) {
      if (token) {
        includes.push(token);
      }
    }
  }
  return includes;
}
function collectSshConfigHosts(rootPath = "", deps = {}) {
  const readFile = deps.readFile || ((p2) => {
    try {
      return fs13.readFileSync(p2, "utf8");
    } catch {
      return null;
    }
  });
  const homeDir = deps.homeDir || os4.homedir();
  const root = rootPath || path13.join(homeDir, ".ssh", "config");
  const sshDir = path13.join(homeDir, ".ssh");
  const out = [];
  const seen = /* @__PURE__ */ new Set();
  const visited = /* @__PURE__ */ new Set();
  const resolveIncludePath = (token) => {
    if (token.startsWith("~/")) {
      return path13.join(homeDir, token.slice(2));
    }
    if (path13.isAbsolute(token)) {
      return token;
    }
    return path13.join(sshDir, token);
  };
  const walk = (filePath, depth) => {
    if (depth > 8 || visited.has(filePath)) {
      return;
    }
    visited.add(filePath);
    const text = readFile(filePath);
    if (text == null) {
      return;
    }
    for (const host of parseSshConfigHosts(text)) {
      if (!seen.has(host)) {
        seen.add(host);
        out.push(host);
      }
    }
    for (const token of parseSshConfigIncludes(text)) {
      const target2 = resolveIncludePath(token);
      const expanded = deps.globSync ? deps.globSync(target2) : [target2];
      for (const p2 of expanded) {
        walk(p2, depth + 1);
      }
    }
  };
  walk(root, 0);
  return out;
}
function parseSshGOutput(text) {
  const out = {
    hostname: null,
    user: null,
    port: null,
    identityFile: null
  };
  for (const rawLine of String(text || "").split("\n")) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    const sp = line.indexOf(" ");
    if (sp === -1) {
      continue;
    }
    const key = line.slice(0, sp).toLowerCase();
    const value = line.slice(sp + 1).trim();
    if (key === "hostname" && !out.hostname) {
      out.hostname = value;
    } else if (key === "user" && !out.user) {
      out.user = value;
    } else if (key === "port" && !out.port) {
      out.port = Number.parseInt(value, 10) || null;
    } else if (key === "identityfile" && !out.identityFile) {
      out.identityFile = value;
    }
  }
  return out;
}

// electron/ssh-connection.ts
import { spawn as spawn3 } from "node:child_process";
import crypto4 from "node:crypto";
import fs14 from "node:fs";
import net from "node:net";
import os5 from "node:os";
import path14 from "node:path";
var DEFAULT_CONNECT_TIMEOUT_MS2 = 15e3;
var DEFAULT_EXEC_TIMEOUT_MS = 2e4;
var DEFAULT_FORWARD_TIMEOUT_MS = 15e3;
var CONTROL_PERSIST_SECONDS = 300;
var _CONTROL_CHAR_RE = /[\x00-\x1f\x7f]/;
function validateSshTarget(host, user, port) {
  if (!host || typeof host !== "string") {
    throw new Error("Unsafe SSH target: host is required.");
  }
  if (host.startsWith("-")) {
    throw new Error(`Unsafe SSH target: host must not start with a dash ("${host}").`);
  }
  if (_CONTROL_CHAR_RE.test(host)) {
    throw new Error("Unsafe SSH target: host contains control characters.");
  }
  if (user && _CONTROL_CHAR_RE.test(user)) {
    throw new Error("Unsafe SSH target: user contains control characters.");
  }
  if (user && user.startsWith("-")) {
    throw new Error(`Unsafe SSH target: user must not start with a dash ("${user}").`);
  }
  const p2 = Number(port);
  if (!Number.isInteger(p2) || p2 < 1 || p2 > 65535) {
    throw new Error(`Unsafe SSH port: ${port} (must be 1-65535).`);
  }
}
function validateKeyPath(keyPath) {
  if (!keyPath) {
    return;
  }
  if (_CONTROL_CHAR_RE.test(keyPath)) {
    throw new Error("Unsafe SSH key path: contains control characters.");
  }
  if (keyPath.startsWith("-")) {
    throw new Error(`Unsafe SSH key path: must not start with a dash ("${keyPath}").`);
  }
}
var _REDACTIONS = [
  [/(HERMES_DASHBOARD_SESSION_TOKEN=)(\S+)/g, "$1<redacted>"],
  [/(X-Hermes-Session-Token["']?\s*[:=]\s*["']?)([^\s"'&]+)/gi, "$1<redacted>"],
  [/(Authorization["']?\s*:\s*Bearer\s+)(\S+)/gi, "$1<redacted>"],
  [/([?&](?:token|ticket)=)([^\s&"']+)/gi, "$1<redacted>"]
];
function redactSecrets(text) {
  let out = String(text == null ? "" : text);
  for (const [re, repl] of _REDACTIONS) {
    out = out.replace(re, repl);
  }
  return out;
}
function controlSocketPath(user, host, port, baseDir, identity = {}) {
  const dir = baseDir || defaultControlDir();
  const keyPathIdentity = path14.normalize(String(identity.keyPath || ""));
  const parts = [
    identity.ownershipId || "",
    identity.scope || "",
    user || "",
    host,
    Number(port),
    keyPathIdentity,
    identity.effectiveConfigFingerprint || ""
  ];
  const id = crypto4.createHash("sha256").update(JSON.stringify(parts)).digest("hex").slice(0, 16);
  return path14.join(dir, `${id}.sock`);
}
function defaultControlDir() {
  if (process.platform === "win32") {
    return path14.join(os5.tmpdir(), "hermes-desktop-ssh");
  }
  return path14.join(os5.homedir(), ".hermes", "desktop-ssh");
}
function baseSshOptions(controlPath, connectTimeoutMs) {
  const connectSecs = Math.max(1, Math.round((connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS2) / 1e3));
  const mux = controlPath ? [
    "-o",
    `ControlPath=${controlPath}`,
    "-o",
    "ControlMaster=auto",
    "-o",
    `ControlPersist=${CONTROL_PERSIST_SECONDS}`
  ] : [];
  return [
    ...mux,
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ExitOnForwardFailure=yes",
    "-o",
    `ConnectTimeout=${connectSecs}`
  ];
}
function hostArgs({ port, keyPath } = {}) {
  const args = [];
  if (port && Number(port) !== 22) {
    args.push("-p", String(port));
  }
  if (keyPath) {
    validateKeyPath(keyPath);
    args.push("-i", keyPath);
  }
  return args;
}
function target(user, host) {
  return user ? `${user}@${host}` : host;
}
function buildExecArgs(conn, remoteCommand, connectTimeoutMs) {
  return [
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    "--",
    target(conn.user, conn.host),
    remoteCommand
  ];
}
function buildControlArgs(conn, op, extra = [], connectTimeoutMs) {
  return [
    "-O",
    op,
    ...extra,
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    "--",
    target(conn.user, conn.host)
  ];
}
function buildMasterArgs(conn, connectTimeoutMs) {
  return [
    "-M",
    "-N",
    "-f",
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    "--",
    target(conn.user, conn.host)
  ];
}
function buildInteractiveSshArgs(conn, remoteCwd, connectTimeoutMs, remoteCommand) {
  const args = [
    "-tt",
    ...baseSshOptions(conn.controlPath, connectTimeoutMs),
    ...hostArgs(conn),
    "--",
    target(conn.user, conn.host)
  ];
  if (remoteCommand) {
    args.push(remoteCommand);
    return args;
  }
  const cwd = String(remoteCwd || "").trim();
  if (cwd) {
    const q2 = `'${cwd.replace(/'/g, `'\\''`)}'`;
    args.push(`cd ${q2} 2>/dev/null; exec "$SHELL" -l`);
  } else {
    args.push('exec "$SHELL" -l');
  }
  return args;
}
function forwardSpec(localPort, remotePort, remoteHost = "127.0.0.1") {
  return `127.0.0.1:${localPort}:${remoteHost}:${remotePort}`;
}
var SSH_ERROR = {
  UNREACHABLE: "unreachable",
  AUTH_FAILED: "auth-failed",
  HOST_KEY_CHANGED: "host-key-changed",
  TIMEOUT: "timeout",
  UNKNOWN: "unknown"
};
function classifySshError(stderr) {
  const text = String(stderr || "");
  if (/REMOTE HOST IDENTIFICATION HAS CHANGED|Host key verification failed|Offending (?:key|ECDSA|RSA|ED25519)/i.test(
    text
  )) {
    return SSH_ERROR.HOST_KEY_CHANGED;
  }
  if (/Permission denied|Too many authentication failures|no matching host key|publickey|password|keyboard-interactive/i.test(
    text
  )) {
    return SSH_ERROR.AUTH_FAILED;
  }
  if (/Could not resolve hostname|Connection refused|Connection timed out|No route to host|Network is unreachable|Operation timed out|port \d+: Connection/i.test(
    text
  )) {
    return SSH_ERROR.UNREACHABLE;
  }
  return SSH_ERROR.UNKNOWN;
}
function sshErrorMessage(kind, conn, stderr) {
  const host = target(conn.user, conn.host);
  switch (kind) {
    case SSH_ERROR.HOST_KEY_CHANGED:
      return `The host key for ${host} has CHANGED since you last connected. This could be a man-in-the-middle attack, or the server was reinstalled. SSH refused to connect. Verify the change is expected, then remove the old key with \`ssh-keygen -R ${conn.host}\` and reconnect.

${String(stderr || "").trim()}`;
    case SSH_ERROR.AUTH_FAILED:
      return `SSH authentication to ${host} failed. Desktop runs ssh non-interactively (BatchMode), so a key requiring a passphrase or 2FA must be loaded into your ssh-agent first (e.g. \`ssh-add ~/.ssh/id_ed25519\`), or set an IdentityFile in ~/.ssh/config. Original error: ${String(stderr || "").trim()}`;
    case SSH_ERROR.UNREACHABLE:
      return `Could not reach ${host} over SSH. Check the host, port, and your network. Original error: ${String(stderr || "").trim()}`;
    case SSH_ERROR.TIMEOUT:
      return `SSH operation to ${host} timed out. The connection may be half-open (e.g. after sleep); reconnecting.`;
    default:
      return `SSH error connecting to ${host}: ${String(stderr || "").trim() || "unknown failure"}`;
  }
}
function runSsh(args, { timeoutMs, spawnFn = spawn3, stdin = "ignore", stdinData } = {}) {
  return new Promise((resolve, reject) => {
    const useStdinPipe = stdinData != null || stdin !== "ignore";
    let child;
    try {
      child = spawnFn("ssh", args, { stdio: [useStdinPipe ? "pipe" : "ignore", "pipe", "pipe"] });
    } catch (error) {
      reject(error);
      return;
    }
    if (stdinData != null && child.stdin) {
      child.stdin.end(stdinData);
    }
    let stdout = "";
    let stderr = "";
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) {
        return;
      }
      settled = true;
      try {
        child.kill("SIGKILL");
      } catch {
      }
      const err = new Error(`ssh timed out after ${timeoutMs}ms`);
      err.kind = SSH_ERROR.TIMEOUT;
      reject(err);
    }, timeoutMs);
    child.stdout?.on("data", (d) => {
      stdout += d.toString();
    });
    child.stderr?.on("data", (d) => {
      stderr += d.toString();
    });
    child.on("error", (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      resolve({ code, stdout, stderr });
    });
  });
}
function stopTunnelChild(child, timeoutMs = 5e3) {
  if (!child || child.exitCode != null || child.signalCode != null) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      child.off?.("exit", onExit);
      child.off?.("error", onError2);
      error ? reject(error) : resolve();
    };
    const onExit = () => finish();
    const onError2 = (error) => finish(error);
    const timer = setTimeout(() => finish(new Error("SSH tunnel did not exit after termination.")), timeoutMs);
    child.once("exit", onExit);
    child.once("error", onError2);
    try {
      if (!child.kill()) {
        finish(new Error("SSH tunnel termination was refused."));
      }
    } catch (error) {
      finish(error);
    }
  });
}
var SshConnection = class {
  host;
  user;
  port;
  keyPath;
  controlPath;
  _spawnFn;
  _log;
  _connectTimeoutMs;
  _execTimeoutMs;
  _forwardTimeoutMs;
  _opened;
  _mux;
  _tunnels;
  constructor(cfg, opts = {}) {
    if (!cfg || !cfg.host) {
      throw new Error("SshConnection requires a host.");
    }
    const port = cfg.port ? Number(cfg.port) : 22;
    validateSshTarget(cfg.host, cfg.user || "", port);
    if (cfg.keyPath) {
      validateKeyPath(cfg.keyPath);
    }
    this.host = cfg.host;
    this.user = cfg.user || "";
    this.port = port;
    this.keyPath = cfg.keyPath || "";
    this._mux = opts.mux ?? process.platform !== "win32";
    this.controlPath = this._mux ? controlSocketPath(this.user, this.host, this.port, opts.controlDir, {
      keyPath: this.keyPath,
      ownershipId: opts.ownershipId,
      scope: opts.scope,
      effectiveConfigFingerprint: opts.effectiveConfigFingerprint
    }) : "";
    this._tunnels = /* @__PURE__ */ new Map();
    this._spawnFn = opts.spawnFn || spawn3;
    this._log = typeof opts.rememberLog === "function" ? opts.rememberLog : () => {
    };
    this._connectTimeoutMs = opts.connectTimeoutMs ?? DEFAULT_CONNECT_TIMEOUT_MS2;
    this._execTimeoutMs = opts.execTimeoutMs ?? DEFAULT_EXEC_TIMEOUT_MS;
    this._forwardTimeoutMs = opts.forwardTimeoutMs ?? DEFAULT_FORWARD_TIMEOUT_MS;
    this._opened = false;
  }
  // Lifecycle logging — ALWAYS through redaction.
  _logLine(msg) {
    this._log(redactSecrets(`[ssh] ${msg}`));
  }
  _fail(stderrOrErr, fallbackKind = SSH_ERROR.UNKNOWN) {
    if (stderrOrErr && stderrOrErr.kind === SSH_ERROR.TIMEOUT) {
      const err2 = new Error(sshErrorMessage(SSH_ERROR.TIMEOUT, this));
      err2.kind = SSH_ERROR.TIMEOUT;
      return err2;
    }
    const stderr = typeof stderrOrErr === "string" ? stderrOrErr : stderrOrErr?.message || "";
    const kind = stderr ? classifySshError(stderr) : fallbackKind;
    const err = new Error(sshErrorMessage(kind, this, stderr));
    err.kind = kind;
    return err;
  }
  // Open the connection. Mux: start the persistent ControlMaster (idempotent —
  // a live master is a no-op). No-mux: there is no master; validate auth +
  // reachability with a one-shot `ssh true` so failures classify identically.
  async open() {
    if (await this.isAlive()) {
      if (!this._mux || await this._verifyMuxChannel()) {
        this._opened = true;
        return;
      }
      this._logLine("existing control master failed exec verification; evicting stale master");
      await this._evictStaleMaster();
    }
    if (!this._mux) {
      this._logLine(`connecting (no-mux) to ${target(this.user, this.host)}:${this.port}`);
      let result2;
      try {
        result2 = await runSsh(buildExecArgs(this, "exit 0", this._connectTimeoutMs), {
          timeoutMs: this._connectTimeoutMs,
          spawnFn: this._spawnFn
        });
      } catch (error) {
        throw this._fail(error, SSH_ERROR.UNREACHABLE);
      }
      if (result2.code !== 0) {
        throw this._fail(result2.stderr, SSH_ERROR.UNREACHABLE);
      }
      this._opened = true;
      this._logLine("connection verified (no-mux; per-operation ssh)");
      return;
    }
    const controlDir = path14.dirname(this.controlPath);
    try {
      fs14.mkdirSync(controlDir, { recursive: true, mode: 448 });
    } catch {
    }
    if (process.platform !== "win32") {
      const st = fs14.lstatSync(controlDir);
      if (st.isSymbolicLink()) {
        throw new Error(`Unsafe SSH control dir: ${controlDir} is a symlink.`);
      }
      if (!st.isDirectory()) {
        throw new Error(`Unsafe SSH control dir: ${controlDir} is not a directory.`);
      }
      if (st.uid !== process.getuid()) {
        throw new Error(`Unsafe SSH control dir: ${controlDir} is owned by uid ${st.uid}, not ${process.getuid()}.`);
      }
      if ((st.mode & 511) !== 448) {
        fs14.chmodSync(controlDir, 448);
      }
    }
    const args = buildMasterArgs(this, this._connectTimeoutMs);
    this._logLine(`opening control master to ${target(this.user, this.host)}:${this.port}`);
    let result;
    try {
      result = await runSsh(args, { timeoutMs: this._connectTimeoutMs, spawnFn: this._spawnFn });
    } catch (error) {
      throw this._fail(error, SSH_ERROR.UNREACHABLE);
    }
    if (result.code !== 0) {
      throw this._fail(result.stderr, SSH_ERROR.UNREACHABLE);
    }
    this._opened = true;
    this._logLine("control master established");
  }
  // Liveness. Mux: `-O check` against the master socket. No-mux: a cheap
  // one-shot exec — "alive" means "we can still authenticate and run".
  async isAlive() {
    if ([...this._tunnels.values()].some((tunnel) => tunnel.alive === false)) {
      return false;
    }
    const args = this._mux ? buildControlArgs(this, "check", [], this._connectTimeoutMs) : buildExecArgs(this, "exit 0", this._connectTimeoutMs);
    try {
      const result = await runSsh(args, { timeoutMs: this._connectTimeoutMs, spawnFn: this._spawnFn });
      return result.code === 0;
    } catch {
      return false;
    }
  }
  // A real exec through the master (`exit 0` works under POSIX shells and
  // cmd.exe); a wedged mux hangs to the timeout.
  async _verifyMuxChannel() {
    try {
      const result = await runSsh(buildExecArgs(this, "exit 0", this._connectTimeoutMs), {
        timeoutMs: this._connectTimeoutMs,
        spawnFn: this._spawnFn
      });
      return result.code === 0;
    } catch {
      return false;
    }
  }
  // -O exit (best-effort) then drop the socket so ControlMaster=auto cannot
  // re-attach to the corpse. (The orphaned master process is left to
  // ControlPersist; a wedged channel can pin it, but without its socket it is
  // inert.)
  async _evictStaleMaster() {
    try {
      await runSsh(buildControlArgs(this, "exit", [], this._connectTimeoutMs), {
        timeoutMs: this._connectTimeoutMs,
        spawnFn: this._spawnFn
      });
    } catch {
    }
    try {
      fs14.unlinkSync(this.controlPath);
    } catch (error) {
      if (error?.code !== "ENOENT") {
        this._logLine(`could not remove stale control socket (${error.code}); a fresh master may not dial`);
      }
    }
  }
  // One-shot remote command over the control connection. Resolves stdout;
  // rejects with a classified error on non-zero exit or timeout.
  async exec(remoteCommand, { timeoutMs, stdinData } = {}) {
    const args = buildExecArgs(this, remoteCommand, this._connectTimeoutMs);
    let result;
    try {
      result = await runSsh(args, {
        timeoutMs: timeoutMs ?? this._execTimeoutMs,
        spawnFn: this._spawnFn,
        ...stdinData != null ? { stdinData } : {}
      });
    } catch (error) {
      throw this._fail(error);
    }
    if (result.code !== 0) {
      throw this._fail(result.stderr);
    }
    return result.stdout;
  }
  // Establish a local→remote forward. Mux: `-O forward` against the master.
  // No-mux: spawn a persistent `ssh -N -L` child that IS the tunnel; ready when
  // the local port accepts. The child dying = tunnel down (isAlive of the
  // backend catches it upstream).
  async forward(localPort, remotePort, remoteHost = "127.0.0.1") {
    const spec = forwardSpec(localPort, remotePort, remoteHost);
    this._logLine(`forwarding 127.0.0.1:${localPort} -> ${remoteHost}:${remotePort}`);
    if (!this._mux) {
      const args2 = [
        ...baseSshOptions("", this._connectTimeoutMs),
        ...hostArgs(this),
        "-v",
        "-N",
        "-L",
        spec,
        "--",
        target(this.user, this.host)
      ];
      const child = this._spawnFn("ssh", args2, { stdio: ["ignore", "ignore", "pipe"] });
      const tunnel = { child, alive: true };
      this._tunnels.set(spec, tunnel);
      let stderr = "";
      let readyConfirmed = false;
      let readyResolve;
      let readyReject;
      const ready = new Promise((resolve, reject) => {
        readyResolve = resolve;
        readyReject = reject;
      });
      const readyPattern = new RegExp(`Local forwarding listening on .* port ${localPort}\\b`);
      child.stderr?.on("data", (d) => {
        if (readyConfirmed) {
          return;
        }
        stderr = `${stderr}${String(d)}`.slice(-16384);
        if (readyPattern.test(stderr)) {
          readyConfirmed = true;
          readyResolve();
        }
      });
      child.on("error", (error) => {
        tunnel.alive = false;
        readyReject(error);
      });
      child.on("exit", (code) => {
        tunnel.alive = false;
        readyReject(new Error(`tunnel process exited with code ${code}`));
      });
      child.on("close", (code) => {
        tunnel.alive = false;
        readyReject(new Error(`tunnel process closed with code ${code}`));
      });
      let readyTimeout;
      try {
        await Promise.race([
          ready,
          new Promise((_2, reject) => {
            readyTimeout = setTimeout(
              () => reject(new Error("tunnel did not confirm local forwarding")),
              this._forwardTimeoutMs
            );
          })
        ]);
      } catch (error) {
        try {
          await stopTunnelChild(child);
          this._tunnels.delete(spec);
        } catch (stopError) {
          throw this._fail(stopError, SSH_ERROR.UNKNOWN);
        }
        throw this._fail(stderr || error, SSH_ERROR.UNKNOWN);
      } finally {
        clearTimeout(readyTimeout);
      }
      return;
    }
    const args = buildControlArgs(this, "forward", ["-L", spec], this._connectTimeoutMs);
    let result;
    try {
      result = await runSsh(args, { timeoutMs: this._forwardTimeoutMs, spawnFn: this._spawnFn });
    } catch (error) {
      throw this._fail(error);
    }
    if (result.code !== 0) {
      throw this._fail(result.stderr);
    }
  }
  // Cancel a previously-established forward. Best-effort: a failure here is
  // logged but not thrown (close tears everything down anyway).
  async cancelForward(localPort, remotePort, remoteHost = "127.0.0.1") {
    const spec = forwardSpec(localPort, remotePort, remoteHost);
    if (!this._mux) {
      const tunnel = this._tunnels.get(spec);
      if (tunnel) {
        await stopTunnelChild(tunnel.child);
        this._tunnels.delete(spec);
        this._logLine(`cancelled forward 127.0.0.1:${localPort}`);
      }
      return;
    }
    const args = buildControlArgs(this, "cancel", ["-L", spec], this._connectTimeoutMs);
    try {
      await runSsh(args, { timeoutMs: this._forwardTimeoutMs, spawnFn: this._spawnFn });
      this._logLine(`cancelled forward 127.0.0.1:${localPort}`);
    } catch (error) {
      this._logLine(`cancelForward failed (ignored): ${error.message}`);
    }
  }
  // Tear down. Mux: exit the master (drops every forward with it). No-mux:
  // kill the tunnel children. Best-effort; never throws.
  async close() {
    if (!this._opened) {
      return;
    }
    if (!this._mux) {
      for (const [spec, tunnel] of this._tunnels) {
        await stopTunnelChild(tunnel.child);
        this._tunnels.delete(spec);
      }
      this._opened = false;
      this._logLine("connection closed (no-mux tunnels killed)");
      return;
    }
    const args = buildControlArgs(this, "exit", [], this._connectTimeoutMs);
    try {
      const result = await runSsh(args, { timeoutMs: this._connectTimeoutMs, spawnFn: this._spawnFn });
      if (result.code !== 0) {
        throw this._fail(result.stderr);
      }
      this._logLine("control master closed");
    } catch (error) {
      this._logLine(`close failed; removing control socket: ${error.message}`);
      try {
        fs14.unlinkSync(this.controlPath);
      } catch {
      }
    }
    this._opened = false;
  }
};
function pickLocalPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address();
      server.close(() => resolve(port));
    });
  });
}
function createSshProbeConnection(config, options = {}) {
  return new SshConnection(config, { ...options, mux: false });
}

// electron/stream-throttle.ts
var RETHROTTLE_DELAY_MS = 5e3;
function createStreamThrottle(timers = { clearTimeout: (handle) => clearTimeout(handle), setTimeout }, delayMs = RETHROTTLE_DELAY_MS) {
  const windows = /* @__PURE__ */ new Set();
  let unthrottled = false;
  let trailing = null;
  function apply(win) {
    if (win.isDestroyed()) {
      windows.delete(win);
      return;
    }
    const contents = win.webContents;
    if (!contents || contents.isDestroyed()) {
      return;
    }
    try {
      contents.setBackgroundThrottling(!unthrottled);
    } catch {
    }
  }
  function applyAll() {
    for (const win of windows) {
      apply(win);
    }
  }
  return {
    isUnthrottled: () => unthrottled,
    register(win) {
      windows.add(win);
      win.on?.("closed", () => windows.delete(win));
      apply(win);
    },
    update(busy) {
      if (busy) {
        if (trailing !== null) {
          timers.clearTimeout(trailing);
          trailing = null;
        }
        if (!unthrottled) {
          unthrottled = true;
          applyAll();
        }
        return;
      }
      if (!unthrottled || trailing !== null) {
        return;
      }
      trailing = timers.setTimeout(() => {
        trailing = null;
        unthrottled = false;
        applyAll();
      }, delayMs);
    }
  };
}

// electron/titlebar-overlay-width.ts
var OVERLAY_FALLBACK_WIDTH = 144;
function nativeOverlayWidth({ isWindows = false, isWsl = false, isMac = false } = {}) {
  if (isMac) {
    return 0;
  }
  return OVERLAY_FALLBACK_WIDTH;
}
var MACOS_TAHOE_DARWIN_MAJOR = 25;
function macTitleBarOverlayHeight({ darwinMajor = 0, titlebarHeight = 0 } = {}) {
  return darwinMajor >= MACOS_TAHOE_DARWIN_MAJOR ? 0 : titlebarHeight;
}

// electron/update-count.ts
function shouldCountCommits({ isShallow, hasMergeBase }) {
  return !(isShallow && !hasMergeBase);
}
function resolveBehindCount({ countStr, currentSha, targetSha, isShallow, hasMergeBase }) {
  if (!shouldCountCommits({ isShallow, hasMergeBase })) {
    if (currentSha && targetSha && currentSha === targetSha) {
      return 0;
    }
    return 1;
  }
  return Number.parseInt(countStr, 10) || 0;
}

// electron/update-gate.ts
function updateGateReason(deps) {
  if (deps.hasLiveMarker()) {
    return "marker";
  }
  if (deps.isUpdateInFlight()) {
    return "update-in-flight";
  }
  return null;
}
async function waitForUpdateClearance(deps, options) {
  const now = options.now || Date.now;
  const sleep2 = options.sleep || ((ms) => new Promise((r2) => setTimeout(r2, ms)));
  let reason = updateGateReason(deps);
  if (!reason) {
    return "clear";
  }
  const deadline = now() + options.timeoutMs;
  while (reason && now() < deadline) {
    if (options.onWaitTick) {
      await options.onWaitTick(reason);
    }
    await sleep2(options.pollMs);
    reason = updateGateReason(deps);
  }
  return reason ? "timeout" : "finished";
}

// electron/update-marker.ts
import fs15 from "fs";
import path15 from "path";
var UPDATE_MARKER_MAX_AGE_MS = 20 * 60 * 1e3;
function markerPath(hermesHome) {
  return path15.join(hermesHome, ".hermes-update-in-progress");
}
function isPidAlive(pid, kill = process.kill.bind(process)) {
  if (!Number.isInteger(pid) || pid <= 0) {
    return false;
  }
  try {
    kill(pid, 0);
    return true;
  } catch (err) {
    return Boolean(err && err.code === "EPERM");
  }
}
function readLiveUpdateMarker(hermesHome, {
  kill,
  now = Date.now,
  maxAgeMs = UPDATE_MARKER_MAX_AGE_MS
} = {}) {
  const file = markerPath(hermesHome);
  let raw;
  try {
    raw = fs15.readFileSync(file, "utf8");
  } catch {
    return null;
  }
  const [pidLine, startedLine] = String(raw).split("\n");
  const pid = Number.parseInt((pidLine || "").trim(), 10);
  const startedAt = Number.parseInt((startedLine || "").trim(), 10);
  const ageMs = Number.isFinite(startedAt) ? now() - startedAt * 1e3 : Infinity;
  const alive = Number.isInteger(pid) && isPidAlive(pid, kill);
  if (!alive || ageMs > maxAgeMs) {
    try {
      fs15.unlinkSync(file);
    } catch {
    }
    return null;
  }
  return { pid, ageMs };
}
function writeUpdateMarker(hermesHome, pid, { now = Date.now } = {}) {
  const file = markerPath(hermesHome);
  const startedAt = Math.floor(now() / 1e3);
  try {
    fs15.writeFileSync(file, `${pid}
${startedAt}
`, "utf8");
  } catch {
  }
}
function updateHandoffConflict(hermesHome, opts = {}) {
  const owner = readLiveUpdateMarker(hermesHome, opts);
  if (!owner) {
    return null;
  }
  const mins = Math.floor(owner.ageMs / 6e4);
  const secs = Math.floor(owner.ageMs % 6e4 / 1e3);
  const elapsed = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  return {
    pid: owner.pid,
    ageMs: owner.ageMs,
    message: `An update is already running (PID ${owner.pid}, started ${elapsed} ago). Wait for it to finish, then try again.`
  };
}

// electron/update-rebuild.ts
function shouldRetryRebuild(code) {
  return code !== 0;
}
async function runRebuildWithRetry(rebuild) {
  let result = await rebuild(0);
  if (shouldRetryRebuild(result.code)) {
    result = await rebuild(1);
  }
  return result;
}

// electron/update-relaunch.ts
import path16 from "node:path";
function unpackedDirName(platform) {
  if (platform === "darwin") {
    return "mac-unpacked";
  }
  if (platform === "win32") {
    return "win-unpacked";
  }
  return "linux-unpacked";
}
function resolveUnpackedRelease(execPath, updateRoot, platform) {
  if (!execPath || !updateRoot) {
    return null;
  }
  const releaseDir = path16.join(updateRoot, "apps", "desktop", "release");
  const unpacked = path16.join(releaseDir, unpackedDirName(platform));
  const normalizedExec = path16.resolve(String(execPath));
  const withSep = unpacked.endsWith(path16.sep) ? unpacked : unpacked + path16.sep;
  if (normalizedExec === unpacked || normalizedExec.startsWith(withSep)) {
    return unpacked;
  }
  return null;
}
function decideRelaunchOutcome({ underUnpacked, sandboxOk }) {
  if (!underUnpacked) {
    return "guiSkew";
  }
  if (!sandboxOk) {
    return "manual";
  }
  return "relaunch";
}
function sandboxPreflight(unpackedDir, statSync2) {
  if (!unpackedDir) {
    return { ok: false, reason: "no-unpacked-dir", path: null };
  }
  const sandboxPath = path16.join(unpackedDir, "chrome-sandbox");
  let st;
  try {
    st = statSync2(sandboxPath);
  } catch {
    return { ok: true, reason: "no-sandbox-helper", path: sandboxPath };
  }
  const ownedByRoot = st.uid === 0;
  const hasSetuid = (st.mode & 2048) !== 0;
  if (ownedByRoot && hasSetuid) {
    return { ok: true, reason: "launchable", path: sandboxPath };
  }
  if (!ownedByRoot && !hasSetuid) {
    return { ok: false, reason: "not-root-not-setuid", path: sandboxPath };
  }
  if (!ownedByRoot) {
    return { ok: false, reason: "not-root", path: sandboxPath };
  }
  return { ok: false, reason: "not-setuid", path: sandboxPath };
}
function sandboxFallbackFromEnv(env2, launchArgs) {
  const disable = String(env2 && env2.ELECTRON_DISABLE_SANDBOX || "").trim();
  if (disable === "1" || disable.toLowerCase() === "true") {
    return true;
  }
  if (Array.isArray(launchArgs) && launchArgs.some((a) => a === "--no-sandbox")) {
    return true;
  }
  return false;
}
function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}
var INTERNAL_ARG_PREFIXES = [
  "--type=",
  // renderer/gpu/zygote child markers
  "--user-data-dir=",
  "--enable-features=",
  "--disable-features=",
  "--field-trial-handle=",
  "--enable-logging",
  "--log-file=",
  // NB: --no-sandbox is deliberately NOT stripped — it reflects the user's /
  // environment's SUID-sandbox opt-out (some hardened kernels/containers require
  // it) and is the signal sandboxFallbackFromEnv() uses to allow a relaunch when
  // chrome-sandbox isn't setuid. Dropping it would make exactly that relaunch
  // fail ("quit and never came back").
  "--disable-gpu-sandbox",
  "--lang=",
  "--inspect",
  "--remote-debugging-port="
];
function collectRelaunchArgs(argv) {
  if (!Array.isArray(argv)) {
    return [];
  }
  return argv.filter((arg) => {
    if (typeof arg !== "string" || arg.length === 0) {
      return false;
    }
    return !INTERNAL_ARG_PREFIXES.some(
      (prefix) => prefix.endsWith("=") ? arg.startsWith(prefix) : arg === prefix || arg.startsWith(prefix + "=")
    );
  });
}
var PRESERVED_ENV_KEYS = ["HERMES_HOME", "ELECTRON_DISABLE_SANDBOX"];
var PRESERVED_ENV_PREFIXES = ["HERMES_DESKTOP_"];
function collectRelaunchEnv(env2) {
  const out = {};
  if (!env2 || typeof env2 !== "object") {
    return out;
  }
  for (const [key, value] of Object.entries(env2)) {
    if (value == null) {
      continue;
    }
    if (PRESERVED_ENV_KEYS.includes(key) || PRESERVED_ENV_PREFIXES.some((p2) => key.startsWith(p2))) {
      out[key] = String(value);
    }
  }
  return out;
}
function buildRelaunchScript({ pid, execPath, args, env: env2, cwd }) {
  const exports = Object.entries(env2 || {}).map(([k2, v]) => `export ${k2}=${shellQuote(v)}`).join("\n");
  const quotedArgs = (args || []).map(shellQuote).join(" ");
  const cwdLine = cwd ? `cd ${shellQuote(cwd)} 2>/dev/null || true` : "";
  return `#!/bin/bash
set -u
APP_PID=${Number(pid)}
# Wait up to ~30s for a graceful exit, then SIGKILL: a hung/zombie parent must
# be gone before we relaunch, or the new instance bails on the single-instance
# lock. (#45205)
for _ in $(seq 1 60); do
  kill -0 "$APP_PID" 2>/dev/null || break
  sleep 0.5
done
if kill -0 "$APP_PID" 2>/dev/null; then
  kill -9 "$APP_PID" 2>/dev/null || true
  sleep 0.5
fi
# Self-delete so temp watchers don't accumulate across updates.
rm -f -- "$0" 2>/dev/null || true
${cwdLine}
${exports}
exec ${shellQuote(execPath)}${quotedArgs ? " " + quotedArgs : ""}
`;
}

// electron/update-remote.ts
var OFFICIAL_REPO_HTTPS_URL = "https://github.com/NousResearch/hermes-agent.git";
var OFFICIAL_REPO_CANONICAL = "github.com/nousresearch/hermes-agent";
function canonicalGitHubRemote(url) {
  if (!url) {
    return "";
  }
  let value = String(url).trim();
  if (value.startsWith("git@github.com:")) {
    value = `github.com/${value.slice("git@github.com:".length)}`;
  } else if (value.startsWith("ssh://git@github.com/")) {
    value = `github.com/${value.slice("ssh://git@github.com/".length)}`;
  } else {
    try {
      const parsed = new URL(value);
      if (parsed.hostname && parsed.pathname) {
        value = `${parsed.hostname}${parsed.pathname}`;
      }
    } catch {
    }
  }
  value = value.trim().replace(/\/+$/, "");
  if (value.endsWith(".git")) {
    value = value.slice(0, -4);
  }
  return value.toLowerCase();
}
function isSshRemote(url) {
  const value = String(url || "").trim().toLowerCase();
  return value.startsWith("git@") || value.startsWith("ssh://");
}
function isOfficialSshRemote(url) {
  return isSshRemote(url) && canonicalGitHubRemote(url) === OFFICIAL_REPO_CANONICAL;
}

// electron/updater-process.ts
import { spawn as spawn4 } from "node:child_process";
import { statSync } from "node:fs";
import path17 from "node:path";
var MARKER_SELF_ADOPT_EPOCH_MS = Date.UTC(2026, 6, 31);
function stagedFileExists(candidate) {
  try {
    return statSync(candidate).isFile();
  } catch {
    return false;
  }
}
function stagedFileMtimeMs(candidate) {
  try {
    return statSync(candidate).mtimeMs;
  } catch {
    return null;
  }
}
function resolveStagedUpdaterBinary(hermesHome, deps = {}) {
  const isWindows = deps.isWindows ?? process.platform === "win32";
  if (!isWindows) {
    return null;
  }
  const fileExists2 = deps.fileExists ?? stagedFileExists;
  const candidate = path17.join(hermesHome, "hermes-setup.exe");
  return fileExists2(candidate) ? candidate : null;
}
function stagedUpdaterSupportsPrewrittenMarker(candidate, deps = {}) {
  const mtimeMs = (deps.stagedMtimeMs ?? stagedFileMtimeMs)(candidate);
  return typeof mtimeMs === "number" && Number.isFinite(mtimeMs) && mtimeMs >= MARKER_SELF_ADOPT_EPOCH_MS;
}
function spawnUpdaterProcess(updater, updaterArgs, options, deps = {}) {
  const isWindows = deps.isWindows ?? process.platform === "win32";
  const spawnOptions = hiddenWindowsChildOptions(options, isWindows);
  const child = deps.spawnProcess ? deps.spawnProcess(updater, updaterArgs, spawnOptions) : spawn4(updater, updaterArgs, spawnOptions);
  child.unref();
  return child;
}

// electron/venv-blocker-scan.ts
import { execFile as execFile3 } from "node:child_process";
import fs16 from "node:fs";
import path18 from "node:path";
import { promisify } from "node:util";
var execFileAsync = promisify(execFile3);
var SCAN_TIMEOUT_MS = 15e3;
var SCAN_MODULE = "hermes_cli._scan_venv_blockers";
function parseVenvBlockerScanOutput(raw) {
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { kind: "probe-failure", error: "malformed JSON" };
  }
  if (!parsed || typeof parsed !== "object" || parsed.ok !== true) {
    return { kind: "probe-failure", error: "missing or invalid ok field" };
  }
  if (typeof parsed.blocked !== "boolean") {
    return { kind: "probe-failure", error: "blocked must be a boolean" };
  }
  if (!Array.isArray(parsed.processes)) {
    return { kind: "probe-failure", error: "processes must be an array" };
  }
  const processes = [];
  for (const entry of parsed.processes) {
    if (!entry || typeof entry !== "object") {
      return { kind: "probe-failure", error: "process entry must be an object" };
    }
    const { pid, name, cmdline } = entry;
    if (!Number.isInteger(pid) || pid <= 0) {
      return { kind: "probe-failure", error: "process pid must be a positive integer" };
    }
    if (typeof name !== "string" || name.length === 0) {
      return { kind: "probe-failure", error: "process name must be a non-empty string" };
    }
    if (typeof cmdline !== "string") {
      return { kind: "probe-failure", error: "process cmdline must be a string" };
    }
    processes.push({ pid, name, cmdline });
  }
  if (parsed.blocked && processes.length === 0) {
    return { kind: "probe-failure", error: "blocked is true but process list is empty" };
  }
  if (!parsed.blocked && processes.length > 0) {
    return { kind: "probe-failure", error: "blocked is false but process list is non-empty" };
  }
  return parsed.blocked ? { kind: "blocked", result: { blocked: true, processes } } : { kind: "clear", result: { blocked: false, processes } };
}
async function scanVenvBlockers(updateRoot, execOverride, resolveOverride) {
  const execFn = execOverride || execFileAsync;
  const resolveFn = resolveOverride || resolveVenvPython;
  const venvPython = resolveFn(updateRoot);
  if (!venvPython) {
    return { kind: "probe-failure", error: "venv python not found" };
  }
  let stdout;
  try {
    const proc = await execFn(venvPython, ["-m", SCAN_MODULE], {
      cwd: updateRoot,
      encoding: "utf-8",
      timeout: SCAN_TIMEOUT_MS,
      windowsHide: true
    });
    stdout = String(proc.stdout ?? "");
  } catch (err) {
    const diag = [`exit code ${err.status ?? err.code ?? -1}`];
    if (err.stderr) {
      diag.push(String(err.stderr).slice(0, 200));
    }
    return { kind: "probe-failure", error: diag.join("; ") };
  }
  return parseVenvBlockerScanOutput(stdout);
}
function resolveVenvPython(updateRoot) {
  const isWindows = process.platform === "win32";
  const pythonName = isWindows ? "python.exe" : "python3";
  const scriptsDir = isWindows ? "Scripts" : "bin";
  const candidate = path18.join(updateRoot, "venv", scriptsDir, pythonName);
  try {
    fs16.accessSync(candidate);
    return candidate;
  } catch {
    return null;
  }
}
function formatBlockerMessage(result) {
  const lines = [
    "Update aborted: another Hermes process is using this installation.",
    "",
    "These processes must be stopped before updating:",
    ""
  ];
  for (const proc of result.processes.slice(0, 10)) {
    lines.push(`  PID ${proc.pid}  ${proc.name}  ${proc.cmdline}`);
  }
  if (result.processes.length > 10) {
    lines.push(`  ... and ${result.processes.length - 10} more`);
  }
  lines.push("");
  lines.push(
    "Close the terminal, app, or service owning that process.  If it is a remote backend, stopping it will disconnect remote clients."
  );
  lines.push("Then retry the update.");
  return lines.join("\n");
}
function formatProbeFailedMessage() {
  return "Update aborted: Desktop could not verify the Hermes installation is free.\n\nClose other Hermes windows and terminals, then retry.  If the problem\npersists, run `hermes update` in a terminal for detailed diagnostics.";
}

// electron/vscode-marketplace.ts
import https2 from "node:https";
import zlib from "node:zlib";
var GALLERY_QUERY_URL = "https://marketplace.visualstudio.com/_apis/public/gallery/extensionquery";
var VSIX_ASSET_TYPE = "Microsoft.VisualStudio.Services.VSIXPackage";
var MAX_VSIX_BYTES = 40 * 1024 * 1024;
var MAX_REDIRECTS = 5;
var REQUEST_TIMEOUT_MS = 2e4;
var ID_RE = /^[\w-]+\.[\w-]+$/;
function request(url, { method = "GET", headers = {}, body = null, maxBytes = MAX_VSIX_BYTES } = {}, redirectsLeft = MAX_REDIRECTS) {
  return new Promise((resolve, reject) => {
    const req = https2.request(url, { method, headers }, (res) => {
      const status = res.statusCode ?? 0;
      if (status >= 300 && status < 400 && res.headers.location) {
        if (redirectsLeft <= 0) {
          res.resume();
          reject(new Error("Too many redirects."));
          return;
        }
        const next = new URL(res.headers.location, url).toString();
        res.resume();
        resolve(
          request(
            next,
            { method: "GET", headers: { "User-Agent": headers["User-Agent"] }, maxBytes },
            redirectsLeft - 1
          )
        );
        return;
      }
      if (status < 200 || status >= 300) {
        res.resume();
        reject(new Error(`Request failed (${status}) for ${url}`));
        return;
      }
      const chunks = [];
      let total = 0;
      res.on("data", (chunk) => {
        total += chunk.length;
        if (total > maxBytes) {
          req.destroy();
          reject(new Error("Response exceeded the size limit."));
          return;
        }
        chunks.push(chunk);
      });
      res.on("end", () => resolve(Buffer.concat(chunks)));
    });
    req.on("error", reject);
    req.setTimeout(REQUEST_TIMEOUT_MS, () => req.destroy(new Error("Request timed out.")));
    if (body) {
      req.write(body);
    }
    req.end();
  });
}
async function resolveExtension(id) {
  const json = await queryGallery({
    // FilterType 7 = ExtensionName (the full publisher.extension id).
    filters: [{ criteria: [{ filterType: 7, value: id }], pageNumber: 1, pageSize: 1 }],
    // Flags: IncludeFiles | IncludeVersionProperties | IncludeAssetUri |
    // IncludeCategoryAndTags | IncludeLatestVersionOnly = 914.
    flags: 914
  });
  const extension = json?.results?.[0]?.extensions?.[0];
  if (!extension) {
    throw new Error(`Extension "${id}" was not found on the Marketplace.`);
  }
  const version = extension.versions?.[0];
  if (!version) {
    throw new Error(`Extension "${id}" has no published versions.`);
  }
  const asset = (version.files ?? []).find((file) => file.assetType === VSIX_ASSET_TYPE);
  const vsixUrl = asset?.source;
  if (!vsixUrl) {
    throw new Error(`Could not find a downloadable package for "${id}".`);
  }
  return { displayName: extension.displayName || id, vsixUrl };
}
async function queryGallery(payload, { maxBytes = 4 * 1024 * 1024 } = {}) {
  const body = JSON.stringify(payload);
  const raw = await request(GALLERY_QUERY_URL, {
    method: "POST",
    headers: {
      Accept: "application/json;api-version=3.0-preview.1",
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(body),
      "User-Agent": "Hermes-Desktop"
    },
    body,
    maxBytes
  });
  return JSON.parse(raw.toString("utf8"));
}
function looksLikeIconTheme(extension) {
  const tags = (extension.tags ?? []).map((tag) => String(tag).toLowerCase());
  if (tags.includes("icon-theme") || tags.includes("product-icon-theme")) {
    return true;
  }
  const text = `${extension.displayName ?? ""} ${extension.shortDescription ?? ""}`.toLowerCase();
  return /\b(icon theme|file icons?|product icons?|icon pack|fileicons)\b/.test(text);
}
async function searchMarketplaceThemes(query, limit = 20) {
  const text = String(query || "").trim();
  const pageSize = Math.min(Math.max(Number(limit) || 20, 1), 50);
  const criteria = [
    { filterType: 8, value: "Microsoft.VisualStudio.Code" },
    { filterType: 5, value: "Themes" },
    { filterType: 12, value: "4096" }
    // Exclude unpublished (Unpublished = 0x1000).
  ];
  if (text) {
    criteria.push({ filterType: 10, value: text });
  }
  const json = await queryGallery({
    // Over-fetch so the icon-theme filter below still leaves a full page.
    filters: [{ criteria, pageNumber: 1, pageSize: Math.min(pageSize * 2, 50), sortBy: 4, sortOrder: 0 }],
    // IncludeStatistics (0x100) | IncludeLatestVersionOnly (0x200) | IncludeCategoryAndTags (0x4).
    flags: 772
  });
  const extensions = json?.results?.[0]?.extensions ?? [];
  return extensions.filter((extension) => !looksLikeIconTheme(extension)).slice(0, pageSize).map((extension) => {
    const publisherName = extension.publisher?.publisherName ?? "";
    const installStat = (extension.statistics ?? []).find((stat) => stat.statisticName === "install");
    return {
      extensionId: `${publisherName}.${extension.extensionName}`,
      displayName: extension.displayName || extension.extensionName,
      publisher: extension.publisher?.displayName || publisherName,
      description: extension.shortDescription || "",
      installs: Math.round(installStat?.value ?? 0)
    };
  });
}
function findEndOfCentralDirectory(buf) {
  for (let i2 = buf.length - 22; i2 >= 0; i2--) {
    if (buf.readUInt32LE(i2) === 101010256) {
      return i2;
    }
  }
  throw new Error("Not a valid zip archive (no end-of-central-directory).");
}
function readCentralDirectory(buf) {
  const eocd = findEndOfCentralDirectory(buf);
  const count = buf.readUInt16LE(eocd + 10);
  let offset = buf.readUInt32LE(eocd + 16);
  const records = /* @__PURE__ */ new Map();
  for (let i2 = 0; i2 < count; i2++) {
    if (buf.readUInt32LE(offset) !== 33639248) {
      break;
    }
    const method = buf.readUInt16LE(offset + 10);
    const compressedSize = buf.readUInt32LE(offset + 20);
    const nameLen = buf.readUInt16LE(offset + 28);
    const extraLen = buf.readUInt16LE(offset + 30);
    const commentLen = buf.readUInt16LE(offset + 32);
    const localOffset = buf.readUInt32LE(offset + 42);
    const name = buf.toString("utf8", offset + 46, offset + 46 + nameLen);
    records.set(name, { method, compressedSize, localOffset });
    offset += 46 + nameLen + extraLen + commentLen;
  }
  return records;
}
function extractEntry(buf, record) {
  if (buf.readUInt32LE(record.localOffset) !== 67324752) {
    throw new Error("Corrupt zip: bad local file header.");
  }
  const nameLen = buf.readUInt16LE(record.localOffset + 26);
  const extraLen = buf.readUInt16LE(record.localOffset + 28);
  const dataStart = record.localOffset + 30 + nameLen + extraLen;
  const data = buf.subarray(dataStart, dataStart + record.compressedSize);
  return record.method === 0 ? data.toString("utf8") : zlib.inflateRawSync(data).toString("utf8");
}
function themeEntryName(themePath) {
  const clean = String(themePath).replace(/^\.\//, "").replace(/^\//, "");
  return `extension/${clean}`;
}
function extractThemes(vsixBuffer) {
  const records = readCentralDirectory(vsixBuffer);
  const pkgRecord = records.get("extension/package.json");
  if (!pkgRecord) {
    throw new Error("Package manifest missing from the extension.");
  }
  const pkg = JSON.parse(extractEntry(vsixBuffer, pkgRecord));
  const contributed = pkg?.contributes?.themes;
  if (!Array.isArray(contributed) || contributed.length === 0) {
    return [];
  }
  const themes = [];
  for (const entry of contributed) {
    if (!entry?.path) {
      continue;
    }
    const record = records.get(themeEntryName(entry.path));
    if (!record) {
      continue;
    }
    try {
      themes.push({
        label: entry.label || entry.id || pkg.displayName || pkg.name || "VS Code Theme",
        uiTheme: entry.uiTheme,
        contents: extractEntry(vsixBuffer, record)
      });
    } catch {
    }
  }
  return themes;
}
async function fetchMarketplaceThemes(id) {
  const trimmed2 = String(id || "").trim();
  if (!ID_RE.test(trimmed2)) {
    throw new Error('Expected a Marketplace id like "publisher.extension".');
  }
  const { displayName, vsixUrl } = await resolveExtension(trimmed2);
  const vsix = await request(vsixUrl, { headers: { "User-Agent": "Hermes-Desktop" } });
  const themes = extractThemes(vsix);
  return { extensionId: trimmed2, displayName, themes };
}

// electron/wake-indicator-window.ts
import { pathToFileURL as pathToFileURL2 } from "node:url";
import { BrowserWindow, screen } from "electron";

// electron/wake-indicator.ts
var WAKE_INDICATOR_WINDOW_WIDTH = 176;
var WAKE_INDICATOR_WINDOW_HEIGHT = 52;
var WAKE_INDICATOR_FADE_MS = 500;
var WAKE_INDICATOR_STATES = ["hidden", "detected", "capturing"];
function normalizeWakeIndicatorState(value) {
  return WAKE_INDICATOR_STATES.includes(value) ? value : "hidden";
}
function selectWakeIndicatorDisplay(displays, primary) {
  return displays.find((display) => display.internal === true) ?? primary;
}
function wakeIndicatorWindowBounds(display) {
  return {
    height: WAKE_INDICATOR_WINDOW_HEIGHT,
    width: WAKE_INDICATOR_WINDOW_WIDTH,
    x: Math.round(display.bounds.x + (display.bounds.width - WAKE_INDICATOR_WINDOW_WIDTH) / 2),
    y: Math.round(display.bounds.y)
  };
}

// electron/wake-indicator-window.ts
function createWakeIndicatorWindowController({
  devServer,
  isMac,
  loadWindowUrl: loadWindowUrl2,
  preloadPath,
  rendererIndex,
  wireWindow
}) {
  let hideTimer = null;
  let state = "hidden";
  let window2 = null;
  const url = () => {
    if (devServer) {
      return `${devServer.endsWith("/") ? devServer.slice(0, -1) : devServer}/?win=wake#/`;
    }
    return `${pathToFileURL2(rendererIndex()).toString()}?win=wake#/`;
  };
  const selectedDisplay = () => selectWakeIndicatorDisplay(screen.getAllDisplays(), screen.getPrimaryDisplay());
  const reposition = () => {
    if (!window2 || window2.isDestroyed()) {
      return;
    }
    window2.setBounds(wakeIndicatorWindowBounds(selectedDisplay()));
  };
  const sendState = () => {
    if (!window2 || window2.isDestroyed()) {
      return;
    }
    window2.webContents.send("hermes:wake-indicator:state", state);
  };
  const spawn6 = () => {
    const next = new BrowserWindow({
      ...wakeIndicatorWindowBounds(selectedDisplay()),
      alwaysOnTop: true,
      backgroundColor: "#00000000",
      focusable: false,
      frame: false,
      fullscreenable: false,
      hasShadow: false,
      hiddenInMissionControl: true,
      maximizable: false,
      minimizable: false,
      movable: false,
      resizable: false,
      show: false,
      skipTaskbar: false,
      transparent: true,
      type: "panel",
      webPreferences: {
        backgroundThrottling: false,
        contextIsolation: true,
        devTools: true,
        nodeIntegration: false,
        preload: preloadPath,
        sandbox: true
      }
    });
    next.setAlwaysOnTop(true, "floating");
    next.setHiddenInMissionControl?.(true);
    next.setIgnoreMouseEvents(true, { forward: true });
    try {
      next.setVisibleOnAllWorkspaces(true, {
        skipTransformProcessType: true,
        visibleOnFullScreen: true
      });
    } catch {
    }
    wireWindow(next);
    next.webContents.on("did-finish-load", sendState);
    next.once("ready-to-show", () => {
      if (!next.isDestroyed() && state !== "hidden") {
        next.showInactive();
      }
    });
    next.on("closed", () => {
      if (window2 === next) {
        window2 = null;
      }
    });
    loadWindowUrl2(next, url(), "Wake indicator");
    return next;
  };
  const setState = (value) => {
    if (!isMac) {
      return;
    }
    state = normalizeWakeIndicatorState(value);
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
    if (state === "hidden") {
      sendState();
      hideTimer = setTimeout(() => {
        hideTimer = null;
        if (state === "hidden" && window2 && !window2.isDestroyed()) {
          window2.hide();
        }
      }, WAKE_INDICATOR_FADE_MS);
      return;
    }
    if (!window2 || window2.isDestroyed()) {
      window2 = spawn6();
    } else {
      reposition();
      sendState();
      window2.showInactive();
    }
  };
  const close = () => {
    if (hideTimer) {
      clearTimeout(hideTimer);
      hideTimer = null;
    }
    if (window2 && !window2.isDestroyed()) {
      window2.close();
    }
    window2 = null;
    state = "hidden";
  };
  return {
    close,
    getState: () => state,
    reposition,
    setState
  };
}

// electron/window-state.ts
var DEFAULT_WIDTH = 1220;
var DEFAULT_HEIGHT = 800;
var MIN_WIDTH = 400;
var MIN_HEIGHT = 620;
var MIN_VISIBLE = 48;
var finite = (v) => typeof v === "number" && Number.isFinite(v);
var clamp = (v, lo, hi) => Math.max(lo, Math.min(v, hi));
function sanitizeWindowState(raw) {
  if (!raw || typeof raw !== "object" || !finite(raw.width) || !finite(raw.height)) {
    return null;
  }
  const state = {
    width: Math.max(MIN_WIDTH, Math.round(raw.width)),
    height: Math.max(MIN_HEIGHT, Math.round(raw.height)),
    isMaximized: raw.isMaximized === true
  };
  if (finite(raw.x) && finite(raw.y)) {
    state.x = Math.round(raw.x);
    state.y = Math.round(raw.y);
  }
  return state;
}
function onScreen(bounds, displays) {
  if (!Array.isArray(displays)) {
    return false;
  }
  return displays.some(({ workArea: a } = {}) => {
    if (!a) {
      return false;
    }
    const x2 = Math.min(bounds.x + bounds.width, a.x + a.width) - Math.max(bounds.x, a.x);
    const y2 = Math.min(bounds.y + bounds.height, a.y + a.height) - Math.max(bounds.y, a.y);
    return x2 >= MIN_VISIBLE && y2 >= MIN_VISIBLE;
  });
}
function computeWindowOptions(state, displays) {
  const opts = {
    width: finite(state?.width) ? state.width : DEFAULT_WIDTH,
    height: finite(state?.height) ? state.height : DEFAULT_HEIGHT
  };
  const cap = (Array.isArray(displays) ? displays : []).reduce(
    (m, { workArea: a } = {}) => a && finite(a.width) && finite(a.height) ? { width: Math.max(m.width, a.width), height: Math.max(m.height, a.height) } : m,
    { width: 0, height: 0 }
  );
  if (cap.width && cap.height) {
    opts.width = clamp(opts.width, MIN_WIDTH, cap.width);
    opts.height = clamp(opts.height, MIN_HEIGHT, cap.height);
  }
  if (state && finite(state.x) && finite(state.y) && onScreen({ x: state.x, y: state.y, width: opts.width, height: opts.height }, displays)) {
    opts.x = state.x;
    opts.y = state.y;
  }
  return opts;
}
function debounce(fn, delayMs) {
  let timer = null;
  const debounced = () => {
    clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn();
    }, delayMs);
  };
  debounced.flush = () => {
    clearTimeout(timer);
    timer = null;
    fn();
  };
  return debounced;
}

// electron/windows-hermes-path.ts
import fs17 from "node:fs";
import path19 from "node:path";
function buildPathExtCandidates(pathext, isWindows) {
  if (!isWindows) {
    return [""];
  }
  return [...(pathext || ".COM;.EXE;.BAT;.CMD").split(";").filter(Boolean), ""];
}
function chooseUpdaterArgs(haveRealInstall, branch) {
  return haveRealInstall ? ["--update", "--branch", branch] : ["--repair", "--branch", branch];
}
function getVenvSitePackagesEntries(venvRoot, opts = {}) {
  const entries = [];
  if (!venvRoot) {
    return entries;
  }
  const isWindows = opts.isWindows ?? process.platform === "win32";
  const directoryExists2 = opts.directoryExists ?? ((p2) => {
    try {
      return fs17.statSync(p2).isDirectory();
    } catch {
      return false;
    }
  });
  const readFile = opts.readFile ?? ((p2) => {
    try {
      return fs17.readFileSync(p2, "utf8");
    } catch {
      return void 0;
    }
  });
  if (isWindows) {
    const sitePackages = path19.join(venvRoot, "Lib", "site-packages");
    if (directoryExists2(sitePackages)) {
      entries.push(sitePackages);
    }
    return entries;
  }
  const cfg = readFile(path19.join(venvRoot, "pyvenv.cfg"));
  const version = (() => {
    if (!cfg) {
      return null;
    }
    const match = cfg.match(/^version_info\s*=\s*(\d+\.\d+)/im);
    return match ? match[1].trim() : null;
  })();
  if (version) {
    const sitePackages = path19.join(venvRoot, "lib", `python${version}`, "site-packages");
    if (directoryExists2(sitePackages)) {
      entries.push(sitePackages);
    }
  }
  return entries;
}
function resolveVenvHermesCommand(command, backendArgs, deps) {
  const {
    isWindows,
    isCommandScript: isCommandScript2,
    fileExists: fileExists2,
    directoryExists: directoryExists2,
    canImportHermesCli: canImportHermesCli2,
    getVenvPython: getVenvPython2,
    getVenvSitePackagesEntries: getVenvSitePackagesEntries2,
    buildDesktopBackendEnv: buildDesktopBackendEnv2,
    hermesHome,
    resolvePath,
    dirname,
    basename,
    rememberLog: rememberLog2
  } = deps;
  if (!isWindows || !command || isCommandScript2(command)) {
    return null;
  }
  const resolved = resolvePath(String(command));
  if (!/^hermes(?:\.exe)?$/i.test(basename(resolved))) {
    return null;
  }
  const scriptsDir = dirname(resolved);
  if (basename(scriptsDir).toLowerCase() !== "scripts") {
    return null;
  }
  const venvRoot = dirname(scriptsDir);
  const python = getVenvPython2(venvRoot);
  if (!fileExists2(python)) {
    return null;
  }
  const root = dirname(venvRoot);
  if (!canImportHermesCli2(python, {
    env: {
      PYTHONPATH: [...directoryExists2(root) ? [root] : [], process.env.PYTHONPATH].filter((entry) => Boolean(entry)).join(path19.delimiter)
    }
  })) {
    rememberLog2?.(
      `Ignoring venv Hermes at ${python}: runtime import probe failed (broken/partial venv); falling through to bootstrap.`
    );
    return null;
  }
  return {
    label: `existing Hermes Python at ${python}`,
    command: python,
    args: ["-m", "hermes_cli.main", ...backendArgs],
    bootstrap: false,
    env: buildDesktopBackendEnv2({
      hermesHome,
      pythonPathEntries: [...directoryExists2(root) ? [root] : [], ...getVenvSitePackagesEntries2(venvRoot)],
      venvRoot
    }),
    kind: "python",
    root,
    shell: false
  };
}

// electron/windows-remote-lifecycle.ts
import crypto5 from "node:crypto";
var LOCKFILE_SCHEMA_VERSION2 = 2;
var PROTOCOL_VERSION2 = 1;
var READY_RE2 = /^HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)/gm;
var READY_POLL_INTERVAL_MS2 = 750;
function psLiteral(value) {
  return `'${String(value).replace(/'/g, "''")}'`;
}
function encodedPowerShell(script) {
  return Buffer.from(script, "utf16le").toString("base64");
}
function powerShellCommand(script) {
  return `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand ${encodedPowerShell(script)}`;
}
async function probeWindowsRemote(ssh, explicitHermesPath = "") {
  const explicit = psLiteral(explicitHermesPath);
  const script = [
    '$ErrorActionPreference="Stop"',
    `$explicit=${explicit}`,
    "$hermesHome=$env:HERMES_HOME",
    'if(-not $hermesHome){$hermesHome=Join-Path $env:LOCALAPPDATA "hermes"}',
    "$candidates=@()",
    "if($explicit){$candidates+=$explicit}",
    "$cmd=Get-Command hermes.exe -ErrorAction SilentlyContinue",
    "if($cmd){$candidates+=$cmd.Source}",
    '$candidates+=(Join-Path $hermesHome "hermes-agent\\venv\\Scripts\\hermes.exe")',
    '$candidates+=(Join-Path $HOME "hermes-agent\\.venv\\Scripts\\hermes.exe")',
    "$hermes=$candidates|Where-Object{Test-Path -LiteralPath $_ -PathType Leaf}|Select-Object -First 1",
    'if(-not $hermes){throw "Hermes is not installed on the remote Windows host."}',
    'if($explicit -and $hermes -ne $explicit){throw "The configured Hermes path is not an executable file."}',
    '$python=Join-Path (Split-Path $hermes) "python.exe"',
    'if(-not (Test-Path -LiteralPath $python -PathType Leaf)){throw "The remote Hermes Python runtime was not found."}',
    '[ordered]@{os="Windows";arch=$env:PROCESSOR_ARCHITECTURE;hermesHome=$hermesHome;hermesPath=$hermes;python=$python}|ConvertTo-Json -Compress'
  ].join(";");
  return JSON.parse((await ssh.exec(powerShellCommand(script))).trim());
}
var TRANSPORT_KINDS = /* @__PURE__ */ new Set([
  SSH_ERROR.AUTH_FAILED,
  SSH_ERROR.HOST_KEY_CHANGED,
  SSH_ERROR.TIMEOUT,
  SSH_ERROR.UNREACHABLE
]);
async function detectRemotePlatform(ssh, explicitHermesPath = "") {
  try {
    const output = (await ssh.exec("uname -s; uname -m")).trim().split("\n");
    if (output[0] === "Linux" || output[0] === "Darwin") {
      return { os: output[0], arch: output[1] || "" };
    }
  } catch (error) {
    if (TRANSPORT_KINDS.has(error?.kind)) {
      throw error;
    }
  }
  try {
    return await probeWindowsRemote(ssh, explicitHermesPath);
  } catch (cause) {
    if (TRANSPORT_KINDS.has(cause?.kind)) {
      throw cause;
    }
    const detail = redactSecrets(String(cause?.message || cause || "")).replace(/[\x00-\x1f\x7f]/g, " ").trim();
    const error = new Error(
      `The remote operating system is not supported by Desktop SSH.${detail ? ` (probe: ${detail.slice(0, 300)})` : ""}`
    );
    error.kind = "unsupported-platform";
    error.cause = cause;
    throw error;
  }
}
function helperCommand(runtime, operation, args = []) {
  const argv = [runtime.python, "-m", "hermes_cli.windows_ssh_runtime", operation, ...args];
  const script = [
    '$ErrorActionPreference="Stop"',
    `& ${argv.map(psLiteral).join(" ")}`,
    "if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}"
  ].join(";");
  return powerShellCommand(script);
}
async function helper(ssh, runtime, operation, args = [], stdinData) {
  const output = await ssh.exec(helperCommand(runtime, operation, args), stdinData == null ? {} : { stdinData });
  const lines = String(output || "").replace(/^\uFEFF/, "").trim().split(/\r?\n/).filter(Boolean);
  const parsed = JSON.parse(lines[lines.length - 1] || "null");
  if (parsed?.error) {
    throw new Error(parsed.error);
  }
  return parsed;
}
function fingerprintToken2(token) {
  return crypto5.createHash("sha256").update(String(token || "")).digest("hex").slice(0, 32);
}
function validLock(lock, ownershipId) {
  return Boolean(
    lock && lock.schemaVersion === LOCKFILE_SCHEMA_VERSION2 && lock.protocolVersion === PROTOCOL_VERSION2 && lock.ownershipId === ownershipId && /^[0-9a-f]{16}$/.test(lock.spawnNonce || "") && Number.isInteger(lock.pid) && lock.pid > 0 && /^[0-9]{10,20}$/.test(lock.creationTimeNs || "") && Number.isInteger(lock.port) && lock.port >= 0 && lock.port <= 65535 && /^[0-9a-f]{32}$/.test(lock.tokenFingerprint || "") && typeof lock.hermesPath === "string" && typeof lock.hermesHome === "string"
  );
}
function reusableWindowsLock(lock, state, profile, reuseToken, runtime) {
  return Boolean(
    state.alive && state.owned && lock.port > 0 && lock.profile === profile && reuseToken && lock.tokenFingerprint === fingerprintToken2(reuseToken) && lock.hermesPath === runtime.hermesPath && lock.hermesHome === runtime.hermesHome
  );
}
function assertCurrent(signal) {
  if (signal?.aborted) {
    const error = new Error("SSH bootstrap was cancelled.");
    error.kind = "superseded";
    throw error;
  }
}
async function processState(ssh, runtime, lock) {
  return helper(ssh, runtime, "process-state", [
    String(lock.pid),
    String(lock.creationTimeNs),
    lock.hermesPath,
    lock.spawnNonce
  ]);
}
async function cleanupOwned(ssh, runtime, ownershipId, lock) {
  const attempt = async (fn) => {
    try {
      await fn();
    } catch {
    }
  };
  if (lock) {
    const state = await processState(ssh, runtime, lock);
    if (state.alive && state.owned) {
      await helper(ssh, runtime, "terminate", [
        String(lock.pid),
        String(lock.creationTimeNs),
        lock.hermesPath,
        lock.spawnNonce
      ]);
    }
    if (lock.spawnNonce) {
      await attempt(() => helper(ssh, runtime, "remove-token", [ownershipId, lock.spawnNonce]));
      await attempt(() => helper(ssh, runtime, "remove-log", [ownershipId, lock.spawnNonce]));
    }
  }
  await attempt(() => helper(ssh, runtime, "remove-lock", [ownershipId]));
}
async function waitReady(ssh, runtime, ownershipId, lock, timeoutMs, signal) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    assertCurrent(signal);
    let state;
    try {
      state = await processState(ssh, runtime, lock);
    } catch {
      await new Promise((resolve) => setTimeout(resolve, READY_POLL_INTERVAL_MS2));
      continue;
    }
    if (!state.indeterminate && (!state.alive || !state.owned)) {
      let detail = "";
      try {
        detail = (await helper(ssh, runtime, "read-log", [ownershipId, lock.spawnNonce]))?.content || "";
      } catch {
      }
      const error2 = new Error(
        `Remote Windows backend exited before announcing its port. state=${JSON.stringify(state)} ${detail.slice(-2e3)}`
      );
      error2.kind = "spawn-failed";
      throw error2;
    }
    let content = "";
    try {
      content = (await helper(ssh, runtime, "read-log", [ownershipId, lock.spawnNonce]))?.content || "";
    } catch {
    }
    let port;
    for (const match of content.matchAll(READY_RE2)) {
      port = Number(match[1]);
    }
    if (port) {
      return port;
    }
    await new Promise((resolve) => setTimeout(resolve, READY_POLL_INTERVAL_MS2));
  }
  const error = new Error(`Timed out waiting for the remote Windows backend (${timeoutMs}ms).`);
  error.kind = "ready-timeout";
  throw error;
}
async function connectWindowsRemote(deps) {
  const {
    ssh,
    ownershipId,
    profile = "",
    remoteHermesPath = "",
    reuseToken = "",
    signal,
    pickLocalPort: pickLocalPort2,
    forward,
    cancelForward,
    waitForHermes: waitForHermes2,
    probeReuseProof,
    rememberLog: rememberLog2 = () => {
    },
    readyTimeoutMs = 45e3
  } = deps;
  assertCurrent(signal);
  const runtime = await probeWindowsRemote(ssh, remoteHermesPath);
  const inspection = await helper(ssh, runtime, "inspect", [runtime.hermesPath]);
  if (!inspection.supported) {
    const error = new Error("Update Hermes on the remote Windows host before connecting with Desktop SSH.");
    error.kind = "update-required";
    throw error;
  }
  runtime.hermesPath = inspection.path;
  const hermesVersion = inspection.version || "";
  rememberLog2(`[ssh-lifecycle] remote platform Windows/${runtime.arch}`);
  rememberLog2(`[ssh-lifecycle] located hermes at ${runtime.hermesPath}`);
  const lock = await helper(ssh, runtime, "read-lock", [ownershipId]);
  if (validLock(lock, ownershipId)) {
    const state = await processState(ssh, runtime, lock);
    if (state.indeterminate) {
      const error = new Error("Could not determine the state of the existing remote backend.");
      error.kind = "transient-transport-error";
      throw error;
    }
    const reusable = reusableWindowsLock(lock, state, profile, reuseToken, runtime);
    if (reusable) {
      const localPort2 = await pickLocalPort2();
      await forward(localPort2, lock.port);
      try {
        const baseUrl = `http://127.0.0.1:${localPort2}`;
        const classification = await probeReuseProof(baseUrl, reuseToken, lock.spawnNonce);
        if (classification === "authenticated-ok") {
          return {
            baseUrl,
            token: reuseToken,
            remotePort: lock.port,
            localPort: localPort2,
            pid: lock.pid,
            reused: true,
            platform: { os: "Windows", arch: runtime.arch },
            hermesPath: runtime.hermesPath,
            hermesVersion,
            ownershipId,
            spawnNonce: lock.spawnNonce,
            creationTimeNs: lock.creationTimeNs
          };
        }
        if (classification !== "authenticated-stale") {
          throw new Error("Invalid SSH reuse classification.");
        }
        await cancelForward(localPort2, lock.port);
        await cleanupOwned(ssh, runtime, ownershipId, lock);
      } catch (error) {
        await cancelForward(localPort2, lock.port);
        throw error;
      }
    } else {
      await cleanupOwned(ssh, runtime, ownershipId, lock);
    }
  } else if (lock) {
    await helper(ssh, runtime, "remove-lock", [ownershipId]);
  }
  assertCurrent(signal);
  const token = crypto5.randomBytes(32).toString("hex");
  const spawnNonce = crypto5.randomBytes(8).toString("hex");
  await helper(ssh, runtime, "upload-token", [ownershipId, spawnNonce], token);
  let spawned;
  try {
    spawned = await helper(
      ssh,
      runtime,
      "spawn",
      [],
      JSON.stringify({ ownershipId, spawnNonce, profile, hermesPath: runtime.hermesPath })
    );
  } catch (error) {
    await helper(ssh, runtime, "remove-token", [ownershipId, spawnNonce]);
    throw error;
  }
  const owned = {
    schemaVersion: LOCKFILE_SCHEMA_VERSION2,
    protocolVersion: PROTOCOL_VERSION2,
    ownershipId,
    spawnNonce,
    pid: spawned.pid,
    creationTimeNs: spawned.creationTimeNs,
    port: 0,
    profile,
    hermesPath: runtime.hermesPath,
    hermesHome: runtime.hermesHome,
    tokenFingerprint: fingerprintToken2(token),
    startedAt: (/* @__PURE__ */ new Date()).toISOString()
  };
  let localPort = 0;
  let remotePort = 0;
  try {
    await helper(ssh, runtime, "write-lock", [ownershipId], JSON.stringify(owned));
    remotePort = await waitReady(ssh, runtime, ownershipId, owned, readyTimeoutMs, signal);
    localPort = await pickLocalPort2();
    await forward(localPort, remotePort);
    const baseUrl = `http://127.0.0.1:${localPort}`;
    await waitForHermes2(baseUrl, token);
    assertCurrent(signal);
    await helper(ssh, runtime, "write-lock", [ownershipId], JSON.stringify({ ...owned, port: remotePort }));
    return {
      baseUrl,
      token,
      remotePort,
      localPort,
      pid: spawned.pid,
      reused: false,
      platform: { os: "Windows", arch: runtime.arch },
      hermesPath: runtime.hermesPath,
      hermesVersion,
      ownershipId,
      spawnNonce,
      creationTimeNs: spawned.creationTimeNs
    };
  } catch (error) {
    if (localPort && remotePort) {
      await cancelForward(localPort, remotePort);
    }
    await cleanupOwned(ssh, runtime, ownershipId, owned);
    throw error;
  }
}
function buildWindowsInteractiveCommand(remoteCwd = "") {
  const cwd = String(remoteCwd || "").trim();
  const script = ['$ErrorActionPreference="Stop"'];
  if (cwd) {
    script.push(
      `if(Test-Path -LiteralPath ${psLiteral(cwd)} -PathType Container){Set-Location -LiteralPath ${psLiteral(cwd)}}`
    );
  }
  script.push('$host.UI.RawUI.WindowTitle="Hermes SSH"', "powershell.exe -NoLogo");
  return powerShellCommand(script.join(";"));
}

// electron/windows-sandbox-fallback.ts
import fs18 from "node:fs";
import path20 from "node:path";
var WINDOWS_SANDBOX_MARKER_FILENAME = "windows-sandbox-fallback.json";
var ALL_APPLICATION_PACKAGES_SID = "S-1-15-2-2";
var WINDOWS_SANDBOX_BREAKPOINT_EXIT = -2147483645;
var BOOT_ABORTS_BEFORE_FALLBACK = 2;
function sandboxMarkerPath(userDataDir) {
  return path20.join(String(userDataDir || ""), WINDOWS_SANDBOX_MARKER_FILENAME);
}
function isWindowsSandboxBreakpointExit(exitCode) {
  const n = Number(exitCode);
  if (!Number.isFinite(n)) {
    return false;
  }
  return n === WINDOWS_SANDBOX_BREAKPOINT_EXIT || n >>> 0 === 2147483651;
}
function alreadyHasNoSandbox(argv = [], env2 = process.env) {
  if (Array.isArray(argv) && argv.some((arg) => arg === "--no-sandbox")) {
    return true;
  }
  const disable = String(env2.ELECTRON_DISABLE_SANDBOX || "").trim().toLowerCase();
  return disable === "1" || disable === "true" || disable === "yes" || disable === "on";
}
var FALLBACK_REASONS = ["gpu-breakpoint", "renderer-crash-loop", "boot-loop"];
function parseSandboxMarker(raw) {
  if (!raw || typeof raw !== "object") {
    return null;
  }
  const record = raw;
  const state = record.state;
  if (state !== "booting" && state !== "fallback" && state !== "ok") {
    return null;
  }
  const marker = { state };
  if (typeof record.reason === "string" && FALLBACK_REASONS.includes(record.reason)) {
    marker.reason = record.reason;
  }
  if (typeof record.version === "string" && record.version) {
    marker.version = record.version;
  }
  const aborts = Number(record.bootAborts);
  if (Number.isInteger(aborts) && aborts > 0) {
    marker.bootAborts = aborts;
  }
  if (record.reprobe === true) {
    marker.reprobe = true;
  }
  return marker;
}
function readSandboxMarker(userDataDir, { readFileSync: readFileSync2 = fs18.readFileSync } = {}) {
  try {
    const raw = JSON.parse(readFileSync2(sandboxMarkerPath(userDataDir), "utf8"));
    return parseSandboxMarker(raw);
  } catch {
    return null;
  }
}
function writeSandboxMarker(userDataDir, marker, {
  mkdirSync = fs18.mkdirSync,
  writeFileSync: writeFileSync2 = fs18.writeFileSync
} = {}) {
  const dir = String(userDataDir || "");
  if (!dir) {
    return;
  }
  mkdirSync(dir, { recursive: true });
  writeFileSync2(sandboxMarkerPath(dir), `${JSON.stringify(marker)}
`, "utf8");
}
function decideWindowsSandboxLaunch(options = {}) {
  const appVersion = String(options.appVersion || "");
  if ((options.platform ?? process.platform) !== "win32") {
    return { enable: false, reason: null, nextMarker: { state: "booting" } };
  }
  const argv = options.argv ?? process.argv;
  const env2 = options.env ?? process.env;
  const marker = options.marker ?? null;
  if (alreadyHasNoSandbox(argv, env2)) {
    const nextMarker = marker?.state === "fallback" ? marker : { state: "booting" };
    return { enable: true, reason: "already-enabled", nextMarker };
  }
  if (marker?.state === "fallback") {
    if (marker.version && appVersion && marker.version !== appVersion) {
      return {
        enable: false,
        reason: null,
        nextMarker: { state: "booting", reprobe: true, bootAborts: 0 }
      };
    }
    return {
      enable: true,
      reason: "sticky-fallback",
      nextMarker: { ...marker, version: marker.version || appVersion || void 0 }
    };
  }
  if (marker?.state === "booting") {
    const abortsObserved = (marker.bootAborts ?? 0) + 1;
    if (marker.reprobe) {
      return {
        enable: true,
        reason: "reprobe-failed",
        nextMarker: fallbackMarker("boot-loop", appVersion)
      };
    }
    if (abortsObserved >= BOOT_ABORTS_BEFORE_FALLBACK) {
      return {
        enable: true,
        reason: "boot-loop",
        nextMarker: fallbackMarker("boot-loop", appVersion)
      };
    }
    return {
      enable: false,
      reason: null,
      nextMarker: { state: "booting", bootAborts: abortsObserved }
    };
  }
  return { enable: false, reason: null, nextMarker: { state: "booting" } };
}
function fallbackMarker(reason, appVersion) {
  const marker = { state: "fallback", reason };
  if (appVersion) {
    marker.version = appVersion;
  }
  return marker;
}
function markerAfterSuccessfulBoot(options) {
  if (!options.fallbackActive) {
    return { state: "ok" };
  }
  return fallbackMarker(options.reason ?? "boot-loop", options.appVersion);
}
function shouldAttemptAclRepair(marker) {
  return marker?.state === "booting" || marker?.state === "fallback";
}
function buildIcaclsGrantArgs(targetDir) {
  return [String(targetDir), "/grant", `*${ALL_APPLICATION_PACKAGES_SID}:(OI)(CI)(RX)`, "/T", "/C", "/Q"];
}
function grantAllApplicationPackagesAcl(targetDir, {
  platform = process.platform,
  execFileSync: execFileSync7
} = {}) {
  if (platform !== "win32") {
    return { ok: false };
  }
  const dir = String(targetDir || "").trim();
  if (!dir || typeof execFileSync7 !== "function") {
    return { ok: false, error: "missing-target-or-exec" };
  }
  try {
    execFileSync7("icacls", buildIcaclsGrantArgs(dir), {
      windowsHide: true,
      timeout: 3e4,
      stdio: "ignore"
    });
    return { ok: true };
  } catch (error) {
    return {
      ok: false,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}
function shouldRelaunchForGpuSandboxCrash(options) {
  if ((options.platform ?? process.platform) !== "win32") {
    return false;
  }
  if (options.alreadyNoSandbox || options.relaunchAttempted) {
    return false;
  }
  const type = String(options.details?.type || "").toLowerCase();
  if (type !== "gpu") {
    return false;
  }
  return isWindowsSandboxBreakpointExit(options.details?.exitCode);
}
function shouldRelaunchForRendererSandboxCrashLoop(options) {
  if ((options.platform ?? process.platform) !== "win32") {
    return false;
  }
  if (options.alreadyNoSandbox || options.relaunchAttempted) {
    return false;
  }
  if (String(options.reason || "") !== "crashed") {
    return false;
  }
  return isWindowsSandboxBreakpointExit(options.exitCode);
}
function buildNoSandboxRelaunchArgs(argv) {
  const args = (Array.isArray(argv) ? argv : []).filter((arg) => arg !== "--no-sandbox");
  args.push("--no-sandbox");
  return args;
}

// electron/windows-system-ca.ts
function installWindowsSystemCaTrust(tlsApi, platform = process.platform) {
  if (platform !== "win32") {
    return {
      applied: false,
      systemCertificateCount: 0,
      totalCertificateCount: 0
    };
  }
  try {
    const defaultCertificates = tlsApi.getCACertificates("default");
    const systemCertificates = tlsApi.getCACertificates("system");
    if (systemCertificates.length === 0) {
      return {
        applied: false,
        systemCertificateCount: 0,
        totalCertificateCount: defaultCertificates.length
      };
    }
    const certificates = [...defaultCertificates, ...systemCertificates];
    tlsApi.setDefaultCACertificates(certificates);
    return {
      applied: true,
      systemCertificateCount: systemCertificates.length,
      totalCertificateCount: certificates.length
    };
  } catch (error) {
    return {
      applied: false,
      systemCertificateCount: 0,
      totalCertificateCount: 0,
      error: error instanceof Error ? error.message : String(error)
    };
  }
}

// electron/windows-user-env.ts
import { execFileSync as execFileSync4 } from "node:child_process";
function parseRegQueryValue(stdout, name) {
  if (!stdout || !name) {
    return null;
  }
  const typePattern = /^(\S+)\s+(?:REG_SZ|REG_EXPAND_SZ|REG_MULTI_SZ|REG_DWORD|REG_QWORD|REG_BINARY|REG_NONE)\s+(.*)$/;
  for (const rawLine of String(stdout).split(/\r?\n/)) {
    const line = rawLine.trim();
    const match = line.match(typePattern);
    if (match && match[1].toLowerCase() === name.toLowerCase()) {
      return match[2];
    }
  }
  return null;
}
function expandWindowsEnvRefs(value, env2 = process.env) {
  if (!value) {
    return value;
  }
  return value.replace(/%([^%]+)%/g, (whole, name) => {
    const key = Object.keys(env2).find((k2) => k2.toUpperCase() === String(name).toUpperCase());
    return key != null && env2[key] != null ? env2[key] : whole;
  });
}
function readWindowsUserEnvVar(name, {
  platform = process.platform,
  env: env2 = process.env,
  exec = execFileSync4
} = {}) {
  if (platform !== "win32" || !name) {
    return null;
  }
  let stdout;
  try {
    stdout = exec("reg", ["query", "HKCU\\Environment", "/v", name], {
      encoding: "utf8",
      windowsHide: true,
      timeout: 5e3
    });
  } catch {
    return null;
  }
  const raw = parseRegQueryValue(stdout, name);
  if (raw == null) {
    return null;
  }
  const expanded = expandWindowsEnvRefs(raw, env2).trim();
  return expanded || null;
}

// electron/workspace-cwd.ts
import path21 from "node:path";
function isPackagedInstallPath(dir, { installRoots, isPackaged }) {
  if (!isPackaged || !dir) {
    return false;
  }
  let resolved;
  try {
    resolved = path21.resolve(String(dir));
  } catch {
    return false;
  }
  const roots = new Set((installRoots ?? []).filter(Boolean).map((candidate) => path21.resolve(String(candidate))));
  for (const root of roots) {
    if (resolved === root) {
      return true;
    }
    const rel = path21.relative(root, resolved);
    if (rel && !rel.startsWith("..") && !path21.isAbsolute(rel)) {
      return true;
    }
  }
  return false;
}

// electron/wsl-clipboard-image.ts
import { execFileSync as execFileSync5 } from "node:child_process";
var PS_SCRIPT = [
  "Add-Type -AssemblyName System.Windows.Forms,System.Drawing",
  "$img = [System.Windows.Forms.Clipboard]::GetImage()",
  "if ($null -eq $img) { exit 0 }",
  "$ms = New-Object System.IO.MemoryStream",
  "$img.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)",
  "[Console]::Out.Write([System.Convert]::ToBase64String($ms.ToArray()))"
].join("\n");
function encodePowerShellCommand(script) {
  return Buffer.from(String(script), "utf16le").toString("base64");
}
function powershellCandidates() {
  return ["powershell.exe", "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"];
}
function decodeClipboardImageBase64(stdout) {
  const b64 = String(stdout || "").trim();
  if (!b64) {
    return null;
  }
  let buffer;
  try {
    buffer = Buffer.from(b64, "base64");
  } catch {
    return null;
  }
  const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  if (buffer.length < PNG_SIGNATURE.length || !buffer.subarray(0, PNG_SIGNATURE.length).equals(PNG_SIGNATURE)) {
    return null;
  }
  return buffer;
}
function readWslWindowsClipboardImage({
  exec = execFileSync5,
  candidates = powershellCandidates()
} = {}) {
  const encoded = encodePowerShellCommand(PS_SCRIPT);
  for (const ps of candidates) {
    try {
      const stdout = exec(
        ps,
        ["-NoProfile", "-NonInteractive", "-STA", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        {
          encoding: "utf8",
          windowsHide: true,
          timeout: 8e3,
          // A 4K screenshot base64s to a few MB; give stdout generous headroom.
          maxBuffer: 64 * 1024 * 1024,
          // PowerShell writes progress/CLIXML noise to stderr — ignore it.
          stdio: ["ignore", "pipe", "ignore"]
        }
      );
      const decoded = decodeClipboardImageBase64(stdout);
      if (decoded) {
        return decoded;
      }
      if (String(stdout || "").trim() === "") {
        return null;
      }
    } catch {
    }
  }
  return null;
}

// electron/zoom.ts
var ZOOM_STORAGE_KEY = "hermes:desktop:zoomLevel";
var ZOOM_FACTOR_BASE = 1.2;
var MIN_ZOOM_LEVEL = -9;
var MAX_ZOOM_LEVEL = 9;
var ZOOM_STEP = 0.1;
var DEFAULT_ZOOM_LEVEL = Math.log(0.9) / Math.log(ZOOM_FACTOR_BASE);
function clampZoomLevel(value) {
  if (!Number.isFinite(value)) {
    return DEFAULT_ZOOM_LEVEL;
  }
  return Math.min(Math.max(value, MIN_ZOOM_LEVEL), MAX_ZOOM_LEVEL);
}
function zoomLevelToPercent(level) {
  return Math.round(Math.pow(ZOOM_FACTOR_BASE, clampZoomLevel(level)) * 100);
}
function percentToZoomLevel(percent) {
  if (!Number.isFinite(percent) || percent <= 0) {
    return DEFAULT_ZOOM_LEVEL;
  }
  return clampZoomLevel(Math.log(percent / 100) / Math.log(ZOOM_FACTOR_BASE));
}
function applyZoomLevel(webContents, level) {
  const clamped = clampZoomLevel(level);
  webContents.setZoomLevel(clamped);
  webContents.send("hermes:zoom:changed", { level: clamped, percent: zoomLevelToPercent(clamped) });
  return clamped;
}
var ZOOM_RESIZE_REASSERT_DELAY_MS = 100;
function zoomReassertWindowEvents(platform = process.platform) {
  return platform === "linux" ? ["show", "restore", "resize", "move"] : ["show", "restore", "resized", "moved"];
}
function installZoomReassertOnWindowEvents(win, reassert, platform = process.platform) {
  if (!win?.on) {
    return;
  }
  let resizeTimer;
  for (const event of zoomReassertWindowEvents(platform)) {
    win.on(event, () => {
      if (win.isDestroyed?.()) {
        return;
      }
      if (event !== "resize" && event !== "move") {
        reassert();
        return;
      }
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (!win.isDestroyed?.()) {
          reassert();
        }
      }, ZOOM_RESIZE_REASSERT_DELAY_MS);
    });
  }
}
var ZOOM_WINDOW_CONFIG = {
  chat: { zoom: true },
  petOverlay: { zoom: false },
  quickEntry: { zoom: false },
  wakeIndicator: { zoom: false }
};
function zoomWiringForWindowKind(kind) {
  return ZOOM_WINDOW_CONFIG[kind] ?? ZOOM_WINDOW_CONFIG.chat;
}

// electron/main.ts
var USER_DATA_OVERRIDE = process.env.HERMES_DESKTOP_USER_DATA_DIR;
if (USER_DATA_OVERRIDE) {
  const resolvedUserData = path22.resolve(USER_DATA_OVERRIDE);
  fs19.mkdirSync(resolvedUserData, { recursive: true });
  app2.setPath("userData", resolvedUserData);
}
var DEV_SERVER = process.env.HERMES_DESKTOP_DEV_SERVER;
var IS_PACKAGED = app2.isPackaged || Boolean(true);
var IS_MAC = process.platform === "darwin";
var IS_WINDOWS3 = process.platform === "win32";
var IS_WSL = isWslEnvironment();
var DARWIN_MAJOR = IS_MAC ? Number.parseInt(os6.release(), 10) || 0 : 0;
var APP_ROOT = app2.getAppPath();
var PRELOAD_PATH = path22.join(APP_ROOT, "dist", "electron-preload.js");
var REMOTE_DISPLAY_REASON = detectRemoteDisplay();
if (REMOTE_DISPLAY_REASON) {
  app2.disableHardwareAcceleration();
  app2.commandLine.appendSwitch("disable-gpu-compositing");
  console.log(
    `[hermes] remote display detected (${REMOTE_DISPLAY_REASON}); disabling GPU hardware acceleration to prevent flicker`
  );
}
var DEV_CDP = resolveDevCdpPort({ env: process.env, isPackaged: IS_PACKAGED, devServer: DEV_SERVER });
if (DEV_CDP.port) {
  app2.commandLine.appendSwitch("remote-debugging-port", String(DEV_CDP.port));
  app2.commandLine.appendSwitch("remote-debugging-address", "127.0.0.1");
  console.log(
    `[hermes] renderer debugging on http://127.0.0.1:${DEV_CDP.port} \u2014 anything that can reach it can run code in the renderer. HERMES_DESKTOP_CDP_PORT=off to disable.`
  );
} else {
  const why = describeDevCdpDecision(DEV_CDP);
  if (why) {
    console.warn(`[hermes] ${why}`);
  }
}
if (IS_WSL && !REMOTE_DISPLAY_REASON && fs19.existsSync("/dev/dxg")) {
  app2.commandLine.appendSwitch("ignore-gpu-blocklist");
  app2.commandLine.appendSwitch("enable-gpu-rasterization");
  app2.commandLine.appendSwitch("enable-zero-copy");
  console.log("[hermes] WSL GPU passthrough (/dev/dxg) detected; enabling GPU acceleration");
}
var windowsSandboxFallbackActive = false;
var windowsSandboxFallbackSticky = false;
var windowsSandboxFallbackReason = "boot-loop";
var windowsNoSandboxRelaunchAttempted = false;
if (IS_WINDOWS3) {
  const windowsUserData = app2.getPath("userData");
  const priorMarker = readSandboxMarker(windowsUserData);
  if (shouldAttemptAclRepair(priorMarker)) {
    const exeDir = path22.dirname(process.execPath);
    const acl = grantAllApplicationPackagesAcl(exeDir, { execFileSync: execFileSync6 });
    if (acl.ok) {
      console.log(`[hermes] granted ALL APPLICATION PACKAGES RX on ${exeDir} (#38216)`);
    } else if (acl.error && acl.error !== "missing-target-or-exec") {
      console.warn(`[hermes] AppContainer ACL grant failed on ${exeDir}: ${acl.error}`);
    }
  }
  const sandboxDecision = decideWindowsSandboxLaunch({
    argv: process.argv,
    env: process.env,
    marker: priorMarker,
    appVersion: app2.getVersion()
  });
  windowsSandboxFallbackActive = sandboxDecision.enable;
  windowsSandboxFallbackSticky = sandboxDecision.nextMarker.state === "fallback";
  if (sandboxDecision.nextMarker.state === "fallback" && sandboxDecision.nextMarker.reason) {
    windowsSandboxFallbackReason = sandboxDecision.nextMarker.reason;
  }
  if (sandboxDecision.enable && sandboxDecision.reason !== "already-enabled") {
    app2.commandLine.appendSwitch("no-sandbox");
    process.env.ELECTRON_DISABLE_SANDBOX = "1";
    console.log(
      `[hermes] Windows sandbox fallback enabled (${sandboxDecision.reason}); launching with --no-sandbox (#38216)`
    );
  }
  writeSandboxMarker(windowsUserData, sandboxDecision.nextMarker);
  app2.on("child-process-gone", (_event, details) => {
    if (!shouldRelaunchForGpuSandboxCrash({
      details,
      alreadyNoSandbox: windowsSandboxFallbackActive || alreadyHasNoSandbox(process.argv, process.env),
      relaunchAttempted: windowsNoSandboxRelaunchAttempted
    })) {
      return;
    }
    windowsNoSandboxRelaunchAttempted = true;
    windowsSandboxFallbackActive = true;
    windowsSandboxFallbackSticky = true;
    windowsSandboxFallbackReason = "gpu-breakpoint";
    try {
      writeSandboxMarker(app2.getPath("userData"), fallbackMarker("gpu-breakpoint", app2.getVersion()));
    } catch {
    }
    console.warn(
      `[hermes] Windows GPU sandbox crashed (exit=${details?.exitCode}); relaunching once with --no-sandbox (#38216)`
    );
    try {
      app2.relaunch({ args: buildNoSandboxRelaunchArgs(process.argv.slice(1)) });
      app2.exit(0);
    } catch (error) {
      console.error(`[hermes] --no-sandbox relaunch failed: ${error?.message || error}`);
    }
  });
}
ipcMain.handle("hermes:get-remote-display-reason", () => REMOTE_DISPLAY_REASON);
app2.commandLine.appendSwitch("disable-renderer-backgrounding");
var SOURCE_REPO_ROOT = path22.resolve(APP_ROOT, "../..");
var INSTALL_STAMP_SCHEMA_VERSION = 1;
function loadInstallStamp() {
  const candidates = [
    process.resourcesPath ? path22.join(process.resourcesPath, "install-stamp.json") : null,
    path22.join(APP_ROOT, "build", "install-stamp.json")
  ].filter(Boolean);
  for (const p2 of candidates) {
    try {
      const raw = fs19.readFileSync(p2, "utf8");
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && typeof parsed.commit === "string" && parsed.commit.length >= 7) {
        if (parsed.schemaVersion !== INSTALL_STAMP_SCHEMA_VERSION) {
          console.warn(
            `[hermes] install-stamp.json schemaVersion ${parsed.schemaVersion} != expected ${INSTALL_STAMP_SCHEMA_VERSION}; ignoring`
          );
          continue;
        }
        return Object.freeze({
          schemaVersion: parsed.schemaVersion,
          commit: parsed.commit,
          branch: parsed.branch || null,
          builtAt: parsed.builtAt || null,
          dirty: Boolean(parsed.dirty),
          source: parsed.source || null,
          path: p2
        });
      }
    } catch (e) {
      console.warn(`[hermes] install-stamp.json found at ${p2} , but parsing failed with ${e}`);
    }
  }
  return null;
}
var INSTALL_STAMP = loadInstallStamp();
if (INSTALL_STAMP) {
  console.log(
    `[hermes] install stamp: ${INSTALL_STAMP.commit.slice(0, 12)}${INSTALL_STAMP.branch ? ` (${INSTALL_STAMP.branch})` : ""}${INSTALL_STAMP.dirty ? " [DIRTY]" : ""} from ${INSTALL_STAMP.source || "unknown"}`
  );
} else if (IS_PACKAGED) {
  console.error(
    "[hermes] WARNING: no install-stamp.json found in packaged build. First-launch bootstrap will not have a pinned ref to install."
  );
}
function resolveHermesHome() {
  if (process.env.HERMES_HOME) {
    return normalizeHermesHomeRoot(process.env.HERMES_HOME);
  }
  if (USER_DATA_OVERRIDE) {
    return path22.join(path22.resolve(USER_DATA_OVERRIDE), "hermes-home");
  }
  if (IS_WINDOWS3) {
    const fromRegistry = readWindowsUserEnvVar("HERMES_HOME");
    if (fromRegistry) {
      return normalizeHermesHomeRoot(fromRegistry);
    }
  }
  if (IS_WINDOWS3 && process.env.LOCALAPPDATA) {
    const localappdata = path22.join(process.env.LOCALAPPDATA, "hermes");
    const legacy = path22.join(app2.getPath("home"), ".hermes");
    if (!directoryExists(localappdata) && directoryExists(legacy)) {
      return legacy;
    }
    return localappdata;
  }
  return path22.join(app2.getPath("home"), ".hermes");
}
var HERMES_HOME = resolveHermesHome();
function pathWithHermesManagedNode(...entries) {
  const managed = hermesManagedNodePathEntries(HERMES_HOME).filter(directoryExists);
  return [...managed, ...entries, process.env.PATH].filter(Boolean).join(path22.delimiter);
}
var ACTIVE_HERMES_ROOT = path22.join(HERMES_HOME, "hermes-agent");
var VENV_ROOT = path22.join(ACTIVE_HERMES_ROOT, "venv");
var BOOTSTRAP_COMPLETE_MARKER = path22.join(ACTIVE_HERMES_ROOT, ".hermes-bootstrap-complete");
var BOOTSTRAP_MARKER_SCHEMA_VERSION = 1;
var DESKTOP_CONNECTION_CONFIG_PATH = path22.join(app2.getPath("userData"), "connection.json");
var DESKTOP_INSTALLATION_PATH = path22.join(app2.getPath("userData"), "desktop-installation.json");
var DESKTOP_UPDATE_CONFIG_PATH = path22.join(app2.getPath("userData"), "updates.json");
var DESKTOP_WINDOW_STATE_PATH = path22.join(app2.getPath("userData"), "window-state.json");
var DESKTOP_PROFILE_CONFIG_PATH = path22.join(app2.getPath("userData"), "active-profile.json");
var PROFILE_NAME_RE = /^[a-z0-9][a-z0-9_-]{0,63}$/;
var DEFAULT_UPDATE_BRANCH = "main";
var DESKTOP_LOG_PATH = path22.join(HERMES_HOME, "logs", "desktop.log");
var DESKTOP_LOG_FLUSH_MS = 120;
var DESKTOP_LOG_BUFFER_MAX_CHARS = 64 * 1024;
var DESKTOP_LOG_MAX_BYTES = 10 * 1024 * 1024;
var DESKTOP_LOG_BACKUP_COUNT = 3;
var DESKTOP_LOG_DISCARD_BYTES = DESKTOP_LOG_MAX_BYTES * 4;
var desktopLogBackupPath = (n) => `${DESKTOP_LOG_PATH}.${n}`;
var BOOT_FAKE_MODE = process.env.HERMES_DESKTOP_BOOT_FAKE === "1";
var BOOT_FAKE_ERROR = process.env.HERMES_DESKTOP_BOOT_FAKE_ERROR || "";
var SKIP_QUIT_CONFIRM = process.env.HERMES_DESKTOP_SKIP_QUIT_CONFIRM === "1";
var BOOT_FAKE_STEP_MS = (() => {
  const raw = Number.parseInt(String(process.env.HERMES_DESKTOP_BOOT_FAKE_STEP_MS || ""), 10);
  if (!Number.isFinite(raw) || raw <= 0) {
    return 650;
  }
  return Math.max(120, raw);
})();
var APP_NAME = process.env.HERMES_DESKTOP_APP_NAME || "Hermes";
var TITLEBAR_HEIGHT = 34;
var MACOS_TRAFFIC_LIGHTS_HEIGHT = 14;
var WINDOW_BUTTON_POSITION = {
  x: 24,
  y: TITLEBAR_HEIGHT / 2 - MACOS_TRAFFIC_LIGHTS_HEIGHT / 2
};
var APP_ICON_PATHS = [
  ...IS_WINDOWS3 ? [path22.join(process.resourcesPath ?? "", "icon.ico"), path22.join(APP_ROOT, "assets", "icon.ico")] : [],
  path22.join(APP_ROOT, "public", "apple-touch-icon.png"),
  path22.join(APP_ROOT, "dist", "apple-touch-icon.png"),
  path22.join(unpackedPathFor(APP_ROOT), "dist", "apple-touch-icon.png")
];
var rendererTitleBarTheme = null;
var terminalSessions = /* @__PURE__ */ new Map();
var NATIVE_THEME_CONFIG_PATH = path22.join(app2.getPath("userData"), "native-theme.json");
var THEME_SOURCES = /* @__PURE__ */ new Set(["dark", "light", "system"]);
function readPersistedThemeSource() {
  try {
    const parsed = JSON.parse(fs19.readFileSync(NATIVE_THEME_CONFIG_PATH, "utf8"));
    if (parsed && THEME_SOURCES.has(parsed.themeSource)) {
      return parsed.themeSource;
    }
  } catch {
  }
  return "system";
}
function writePersistedThemeSource(mode) {
  try {
    fs19.mkdirSync(path22.dirname(NATIVE_THEME_CONFIG_PATH), { recursive: true });
    fs19.writeFileSync(NATIVE_THEME_CONFIG_PATH, JSON.stringify({ themeSource: mode }, null, 2), "utf8");
  } catch (error) {
    rememberLog(`[theme] write native theme failed: ${error.message}`);
  }
}
nativeTheme.themeSource = readPersistedThemeSource();
var TRANSLUCENCY_CONFIG_PATH = path22.join(app2.getPath("userData"), "translucency.json");
function clampIntensity(value) {
  const n = Math.round(Number(value));
  return Number.isFinite(n) ? Math.min(100, Math.max(0, n)) : 0;
}
function readPersistedTranslucency() {
  try {
    return clampIntensity(JSON.parse(fs19.readFileSync(TRANSLUCENCY_CONFIG_PATH, "utf8")).intensity);
  } catch {
    return 0;
  }
}
function writePersistedTranslucency(intensity) {
  try {
    fs19.mkdirSync(path22.dirname(TRANSLUCENCY_CONFIG_PATH), { recursive: true });
    fs19.writeFileSync(TRANSLUCENCY_CONFIG_PATH, JSON.stringify({ intensity }, null, 2), "utf8");
  } catch (error) {
    rememberLog(`[translucency] write failed: ${error.message}`);
  }
}
var translucencyIntensity = readPersistedTranslucency();
function windowOpacity() {
  return 1 - translucencyIntensity / 100 * 0.7;
}
function applyWindowTranslucency(win) {
  if (!win || win.isDestroyed() || typeof win.setOpacity !== "function") {
    return;
  }
  try {
    win.setOpacity(windowOpacity());
  } catch (error) {
    rememberLog(`[translucency] apply failed: ${error.message}`);
  }
}
function isHexColor(value) {
  return typeof value === "string" && /^#[0-9a-f]{6}$/i.test(value);
}
function getWindowBackgroundColor() {
  if (rendererTitleBarTheme && isHexColor(rendererTitleBarTheme.background)) {
    return rendererTitleBarTheme.background;
  }
  return nativeTheme.shouldUseDarkColors ? "#111111" : "#f7f7f7";
}
var TITLEBAR_OVERLAY_COLOR = "rgba(1, 0, 0, 0)";
function getTitleBarOverlayOptions() {
  if (IS_MAC) {
    return { height: macTitleBarOverlayHeight({ darwinMajor: DARWIN_MAJOR, titlebarHeight: TITLEBAR_HEIGHT }) };
  }
  if (!IS_WINDOWS3 && IS_WSL) {
    return false;
  }
  return {
    color: TITLEBAR_OVERLAY_COLOR,
    height: TITLEBAR_HEIGHT,
    symbolColor: rendererTitleBarTheme && isHexColor(rendererTitleBarTheme.foreground) ? rendererTitleBarTheme.foreground : nativeTheme.shouldUseDarkColors ? "#f7f7f7" : "#242424"
  };
}
function applyTitleBarOverlay(win) {
  const options = getTitleBarOverlayOptions();
  if (!options || typeof options !== "object") {
    return;
  }
  try {
    win?.setTitleBarOverlay?.(options);
  } catch {
  }
}
var MEDIA_MIME_TYPES = {
  ".avi": "video/x-msvideo",
  ".bmp": "image/bmp",
  ".flac": "audio/flac",
  ".gif": "image/gif",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".m4a": "audio/mp4",
  ".mkv": "video/x-matroska",
  ".mov": "video/quicktime",
  ".mp3": "audio/mpeg",
  ".mp4": "video/mp4",
  ".ogg": "audio/ogg",
  ".opus": "audio/ogg; codecs=opus",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".wav": "audio/wav",
  ".webm": "video/webm",
  ".webp": "image/webp"
};
var PREVIEW_HTML_EXTENSIONS = /* @__PURE__ */ new Set([".html", ".htm"]);
var PREVIEW_WATCH_DEBOUNCE_MS = 120;
var LOCAL_PREVIEW_HOSTS = /* @__PURE__ */ new Set(["0.0.0.0", "127.0.0.1", "::1", "[::1]", "localhost"]);
var TEXT_PREVIEW_MAX_BYTES = 512 * 1024;
var PREVIEW_LANGUAGE_BY_EXT = {
  ".c": "c",
  ".conf": "ini",
  ".cpp": "cpp",
  ".css": "css",
  ".csv": "csv",
  ".go": "go",
  ".graphql": "graphql",
  ".h": "c",
  ".hpp": "cpp",
  ".html": "html",
  ".java": "java",
  ".js": "javascript",
  ".json": "json",
  ".jsx": "jsx",
  ".kt": "kotlin",
  ".lua": "lua",
  ".md": "markdown",
  ".mjs": "javascript",
  ".py": "python",
  ".rb": "ruby",
  ".rs": "rust",
  ".sh": "shell",
  ".sql": "sql",
  ".svg": "xml",
  ".toml": "toml",
  ".ts": "typescript",
  ".tsx": "tsx",
  ".txt": "text",
  ".xml": "xml",
  ".yaml": "yaml",
  ".yml": "yaml",
  ".zsh": "shell"
};
function looksBinary(buffer) {
  if (!buffer.length) {
    return false;
  }
  let suspicious = 0;
  for (const byte of buffer) {
    if (byte === 0) {
      return true;
    }
    if (byte < 32 && byte !== 9 && byte !== 10 && byte !== 13) {
      suspicious += 1;
    }
  }
  return suspicious / buffer.length > 0.12;
}
function previewFileMetadata(filePath, mimeType) {
  let byteSize = 0;
  let binary = false;
  try {
    const stat = fs19.statSync(filePath);
    byteSize = stat.size;
    if (!mimeType.startsWith("image/")) {
      const fd = fs19.openSync(filePath, "r");
      try {
        const sample = Buffer.alloc(Math.min(byteSize, 4096));
        const bytesRead = fs19.readSync(fd, sample, 0, sample.length, 0);
        binary = looksBinary(sample.subarray(0, bytesRead));
      } finally {
        fs19.closeSync(fd);
      }
    }
  } catch {
  }
  return {
    binary,
    byteSize,
    large: byteSize > TEXT_PREVIEW_MAX_BYTES
  };
}
app2.setName(APP_NAME);
if (IS_WINDOWS3) {
  app2.setAppUserModelId("com.nousresearch.hermes");
}
app2.setAboutPanelOptions({
  applicationName: APP_NAME,
  applicationVersion: resolveHermesVersion(),
  copyright: "Copyright \xA9 2026 Nous Research"
});
var MEDIA_PROTOCOL = "hermes-media";
var STREAMABLE_MEDIA_EXTS = /* @__PURE__ */ new Set([
  ".avi",
  ".flac",
  ".m4a",
  ".mkv",
  ".mov",
  ".mp3",
  ".mp4",
  ".ogg",
  ".opus",
  ".wav",
  ".webm"
]);
protocol.registerSchemesAsPrivileged([
  {
    scheme: MEDIA_PROTOCOL,
    privileges: {
      secure: true,
      standard: true,
      stream: true,
      supportFetchAPI: true
    }
  }
]);
function registerMediaProtocol() {
  protocol.handle(MEDIA_PROTOCOL, async (request2) => {
    let resolvedPath;
    try {
      const url = new URL(request2.url);
      const filePath = decodeURIComponent(url.pathname.replace(/^\/+/, ""));
      ({ resolvedPath } = await resolveReadableFileForIpc(filePath, { purpose: "Media stream" }));
    } catch {
      return new Response("Media not found", { status: 404 });
    }
    if (!STREAMABLE_MEDIA_EXTS.has(path22.extname(resolvedPath).toLowerCase())) {
      return new Response("Unsupported media type", { status: 415 });
    }
    return electronNet.fetch(pathToFileURL3(resolvedPath).toString(), {
      bypassCustomProtocolHandlers: true,
      headers: request2.headers
    });
  });
}
var mainWindow = null;
var backendConnectionState = createBackendConnectionState();
var remoteLiveness = new RemoteLivenessTracker();
var remoteRevalidation = new RemoteRevalidationCoordinator();
var softRehomeInProgress = false;
var backendPool = /* @__PURE__ */ new Map();
var POOL_MAX_BACKENDS = Math.max(1, Number(process.env.HERMES_DESKTOP_POOL_MAX) || 3);
var POOL_IDLE_MS = Math.max(6e4, Number(process.env.HERMES_DESKTOP_POOL_IDLE_MS) || 10 * 6e4);
var POOL_KEEPALIVE_FRESH_MS = 9e4;
var poolIdleReaper = null;
var RENDERER_RELOAD_WINDOW_MS = 6e4;
var RENDERER_RELOAD_MAX = 3;
var rendererReloadTimes = [];
var bootstrapFailure = null;
var backendStartFailure = null;
var remoteReauthFailure = null;
var bootstrapAbortController = null;
var bootstrapRepairRequested = false;
var bootstrapRepairAttempt = 0;
var MAX_BOOTSTRAP_REPAIR_SOFT_ATTEMPTS = 3;
var connectionConfigCache = null;
var connectionConfigCacheMtime = null;
var hermesLog = [];
var previewWatchers = /* @__PURE__ */ new Map();
var previewShortcutActive = false;
var desktopLogBuffer = "";
var desktopLogFlushTimer = null;
var desktopLogFlushPromise = Promise.resolve();
var nativeThemeListenerInstalled = false;
var bootProgressState = {
  error: null,
  fakeMode: BOOT_FAKE_MODE,
  message: "Waiting to start Hermes backend",
  phase: "idle",
  progress: 0,
  running: false,
  timestamp: Date.now()
};
function planDesktopLogRotation(size) {
  if (size < DESKTOP_LOG_MAX_BYTES) {
    return [];
  }
  const backups = (n) => Array.from({ length: n }, (_2, i2) => desktopLogBackupPath(i2 + 1));
  if (size > DESKTOP_LOG_DISCARD_BYTES) {
    return [DESKTOP_LOG_PATH, ...backups(DESKTOP_LOG_BACKUP_COUNT)].map((p2) => ["rm", p2]);
  }
  const ops = [["rm", desktopLogBackupPath(DESKTOP_LOG_BACKUP_COUNT)]];
  for (let i2 = DESKTOP_LOG_BACKUP_COUNT - 1; i2 >= 1; i2--) {
    ops.push(["mv", desktopLogBackupPath(i2), desktopLogBackupPath(i2 + 1)]);
  }
  ops.push(["mv", DESKTOP_LOG_PATH, desktopLogBackupPath(1)]);
  return ops;
}
function rotateDesktopLogIfNeededSync() {
  let size;
  try {
    size = fs19.statSync(DESKTOP_LOG_PATH).size;
  } catch {
    return;
  }
  for (const [op, src, dst] of planDesktopLogRotation(size)) {
    try {
      if (op === "rm") {
        fs19.rmSync(src, { force: true });
      } else {
        fs19.renameSync(src, dst);
      }
    } catch {
    }
  }
}
async function rotateDesktopLogIfNeededAsync() {
  let size;
  try {
    size = (await fs19.promises.stat(DESKTOP_LOG_PATH)).size;
  } catch {
    return;
  }
  for (const [op, src, dst] of planDesktopLogRotation(size)) {
    try {
      if (op === "rm") {
        await fs19.promises.rm(src, { force: true });
      } else {
        await fs19.promises.rename(src, dst);
      }
    } catch {
    }
  }
}
function flushDesktopLogBufferSync() {
  if (!desktopLogBuffer) {
    return;
  }
  const chunk = desktopLogBuffer;
  desktopLogBuffer = "";
  try {
    fs19.mkdirSync(path22.dirname(DESKTOP_LOG_PATH), { recursive: true });
    rotateDesktopLogIfNeededSync();
    fs19.appendFileSync(DESKTOP_LOG_PATH, chunk);
  } catch {
  }
}
function flushDesktopLogBufferAsync() {
  if (!desktopLogBuffer) {
    return desktopLogFlushPromise;
  }
  const chunk = desktopLogBuffer;
  desktopLogBuffer = "";
  desktopLogFlushPromise = desktopLogFlushPromise.then(async () => {
    await fs19.promises.mkdir(path22.dirname(DESKTOP_LOG_PATH), { recursive: true });
    await rotateDesktopLogIfNeededAsync();
    await fs19.promises.appendFile(DESKTOP_LOG_PATH, chunk);
  }).catch(() => {
  });
  return desktopLogFlushPromise;
}
function scheduleDesktopLogFlush() {
  if (desktopLogFlushTimer) {
    return;
  }
  desktopLogFlushTimer = setTimeout(() => {
    desktopLogFlushTimer = null;
    void flushDesktopLogBufferAsync();
  }, DESKTOP_LOG_FLUSH_MS);
}
function rememberLog(chunk) {
  const text = String(chunk || "").trim();
  if (!text) {
    return;
  }
  const lines = text.split(/\r?\n/).map((line) => `[hermes] ${line}`);
  hermesLog.push(...lines);
  if (hermesLog.length > 300) {
    hermesLog.splice(0, hermesLog.length - 300);
  }
  desktopLogBuffer += `${lines.join("\n")}
`;
  if (desktopLogBuffer.length >= DESKTOP_LOG_BUFFER_MAX_CHARS) {
    if (desktopLogFlushTimer) {
      clearTimeout(desktopLogFlushTimer);
      desktopLogFlushTimer = null;
    }
    void flushDesktopLogBufferAsync();
    return;
  }
  scheduleDesktopLogFlush();
}
installCrashForensics({ flush: flushDesktopLogBufferSync, log: rememberLog });
function loadWindowUrl(win, url, label) {
  win.loadURL(url).catch((error) => rememberLog(`${label} failed to load: ${describeCrashReason(error)}`));
}
function openExternalUrl(rawUrl) {
  const raw = String(rawUrl || "").trim();
  if (!raw) {
    return false;
  }
  let parsed;
  try {
    parsed = new URL(raw);
  } catch {
    return false;
  }
  if (parsed.protocol === "file:") {
    let localPath;
    try {
      localPath = resolveRequestedPathForIpc(parsed.toString(), { purpose: "Open external file" });
    } catch {
      return false;
    }
    void shell.openPath(localPath).then((error) => {
      if (!error) {
        return;
      }
      rememberLog(`[file] openPath failed: ${error}; revealing in folder instead`);
      try {
        shell.showItemInFolder(localPath);
      } catch (revealError) {
        rememberLog(`[file] showItemInFolder failed: ${revealError.message}`);
      }
    }).catch((error) => rememberLog(`[file] openPath rejected: ${error.message}`));
    return true;
  }
  if (!["http:", "https:", "mailto:"].includes(parsed.protocol)) {
    return false;
  }
  const url = parsed.toString();
  if (IS_WSL) {
    rememberLog(`[link] opening via WSL\u2192Windows: ${url}`);
    const proc = spawn5("cmd.exe", ["/c", "start", '""', url], {
      detached: true,
      stdio: "ignore",
      windowsHide: true
    });
    proc.on("error", (error) => {
      rememberLog(`[link] cmd.exe start failed: ${error.message}; falling back to xdg-open`);
      shell.openExternal(url).catch((fallback) => rememberLog(`[link] xdg-open failed: ${fallback.message}`));
    });
    proc.unref();
    return true;
  }
  shell.openExternal(url).catch((error) => rememberLog(`[link] openExternal failed: ${error.message}`));
  return true;
}
async function openPreviewInBrowser(rawUrl) {
  const raw = String(rawUrl || "").trim();
  if (!raw) {
    return false;
  }
  let parsed;
  try {
    parsed = new URL(raw);
  } catch {
    return false;
  }
  if (parsed.protocol === "file:") {
    let localPath;
    try {
      localPath = resolveRequestedPathForIpc(parsed.toString(), { purpose: "Open preview in browser" });
    } catch {
      return false;
    }
    await shell.openExternal(pathToFileURL3(localPath).toString());
    return true;
  }
  return openExternalUrl(raw);
}
function ensureWslWindowsFonts() {
  if (!IS_WSL) {
    return;
  }
  const fontsDir = ["/mnt/c/Windows/Fonts", "/mnt/c/windows/fonts"].find((candidate) => {
    try {
      return fs19.statSync(candidate).isDirectory();
    } catch {
      return false;
    }
  });
  if (!fontsDir) {
    return;
  }
  try {
    const confDir = path22.join(app2.getPath("home"), ".config", "fontconfig", "conf.d");
    const confPath = path22.join(confDir, "99-hermes-wsl-windows-fonts.conf");
    let existing = "";
    try {
      existing = fs19.readFileSync(confPath, "utf8");
    } catch {
      existing = "";
    }
    if (existing.includes(fontsDir)) {
      return;
    }
    fs19.mkdirSync(confDir, { recursive: true });
    fs19.writeFileSync(
      confPath,
      `<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <dir>${fontsDir}</dir>
</fontconfig>
`
    );
    rememberLog(`[fonts] wired WSL Windows fonts for renderer: ${fontsDir}`);
    const cache = spawn5("fc-cache", ["-f", fontsDir], { detached: true, stdio: "ignore" });
    cache.on("error", () => void 0);
    cache.unref();
  } catch (error) {
    rememberLog(`[fonts] WSL font setup skipped: ${error.message}`);
  }
}
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
function clampBootProgress(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(numeric)));
}
function broadcastBootProgress() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  webContents.send("hermes:boot-progress", bootProgressState);
}
var BOOTSTRAP_LOG_RING_MAX = 500;
var bootstrapState = {
  active: false,
  manifest: null,
  stages: {},
  error: null,
  log: [],
  startedAt: null,
  completedAt: null,
  setupChoice: null,
  unsupportedPlatform: null
};
var firstRunSetupGate = null;
function broadcastBootstrapEvent(ev) {
  if (ev.type === "manifest") {
    bootstrapState.manifest = ev;
    bootstrapState.active = true;
    bootstrapState.setupChoice = null;
    bootstrapState.startedAt = bootstrapState.startedAt || Date.now();
    bootstrapState.stages = {};
    for (const stage of ev.stages || []) {
      bootstrapState.stages[stage.name] = { state: "pending", json: null, durationMs: null, error: null };
    }
  } else if (ev.type === "stage") {
    bootstrapState.stages[ev.name] = {
      state: ev.state,
      durationMs: ev.durationMs ?? null,
      json: ev.json ?? null,
      error: ev.error ?? null
    };
  } else if (ev.type === "log") {
    bootstrapState.log.push({ ts: Date.now(), stage: ev.stage || null, line: ev.line, stream: ev.stream || "stdout" });
    if (bootstrapState.log.length > BOOTSTRAP_LOG_RING_MAX) {
      bootstrapState.log.splice(0, bootstrapState.log.length - BOOTSTRAP_LOG_RING_MAX);
    }
  } else if (ev.type === "complete") {
    bootstrapState.active = false;
    bootstrapState.completedAt = Date.now();
    bootstrapState.error = null;
    bootstrapState.unsupportedPlatform = null;
  } else if (ev.type === "failed") {
    bootstrapState.active = false;
    bootstrapState.error = ev.error || "unknown error";
    bootstrapState.setupChoice = null;
  } else if (ev.type === "unsupported-platform") {
    bootstrapState.active = false;
    bootstrapState.setupChoice = null;
    bootstrapState.unsupportedPlatform = {
      platform: ev.platform,
      activeRoot: ev.activeRoot,
      installCommand: ev.installCommand,
      docsUrl: ev.docsUrl
    };
  } else if (ev.type === "setup-choice") {
    bootstrapState.active = false;
    bootstrapState.error = null;
    bootstrapState.manifest = null;
    bootstrapState.stages = {};
    bootstrapState.setupChoice = ev.active ? {
      platform: ev.platform,
      activeRoot: ev.activeRoot
    } : null;
    bootstrapState.unsupportedPlatform = null;
  } else if (ev.type === "dismissed") {
    resetBootstrapSnapshot();
  }
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  webContents.send("hermes:bootstrap:event", ev);
}
function getBootstrapState() {
  return bootstrapState;
}
function resetBootstrapSnapshot() {
  bootstrapState = {
    active: false,
    manifest: null,
    stages: {},
    error: null,
    log: [],
    startedAt: null,
    completedAt: null,
    setupChoice: null,
    unsupportedPlatform: null
  };
}
function promptFirstRunSetupChoice(backend) {
  broadcastBootstrapEvent({
    type: "setup-choice",
    active: true,
    platform: backend.platform || process.platform,
    activeRoot: backend.activeRoot || ACTIVE_HERMES_ROOT
  });
}
function hideFirstRunSetupChoice() {
  if (bootstrapState.setupChoice) {
    broadcastBootstrapEvent({ type: "setup-choice", active: false });
  }
}
function getFirstRunSetupGate() {
  if (!firstRunSetupGate) {
    firstRunSetupGate = createFirstRunSetupGate({
      hideChoice: hideFirstRunSetupChoice,
      log: rememberLog,
      onStuck: (_backend, stuckAfterMs) => {
        updateBootProgress(
          {
            error: null,
            message: `Still waiting for first-run setup choice after ${Math.round(stuckAfterMs / 1e3)} seconds`,
            phase: "bootstrap.choice",
            progress: 12,
            running: true
          },
          { allowDecrease: true }
        );
      },
      promptChoice: promptFirstRunSetupChoice
    });
  }
  return firstRunSetupGate;
}
async function waitForFirstRunSetupChoice(backend) {
  const gate = getFirstRunSetupGate();
  if (!gate.shouldGate(backend)) {
    return "continue-local";
  }
  updateBootProgress(
    {
      error: null,
      message: "Waiting for first-run setup choice",
      phase: "bootstrap.choice",
      progress: 12,
      running: true
    },
    { allowDecrease: true }
  );
  return gate.wait(backend);
}
function continueFirstRunLocalBootstrap() {
  getFirstRunSetupGate().continueLocal();
}
function abandonFirstRunSetupChoiceForRemoteApply() {
  const gate = getFirstRunSetupGate();
  if (!gate.hasWaiter()) {
    return false;
  }
  const resumedGatedConnection = gate.abandonForRemoteApply();
  if (resumedGatedConnection) {
    broadcastBootstrapEvent({ type: "dismissed" });
  }
  return resumedGatedConnection;
}
function updateBootProgress(update, options = {}) {
  const nextProgressRaw = typeof update.progress === "number" ? clampBootProgress(update.progress) : bootProgressState.progress;
  const nextProgress = options.allowDecrease ? nextProgressRaw : Math.max(bootProgressState.progress, nextProgressRaw);
  bootProgressState = {
    ...bootProgressState,
    ...update,
    error: update.error === void 0 ? bootProgressState.error : update.error,
    fakeMode: BOOT_FAKE_MODE || Boolean(update.fakeMode),
    progress: nextProgress,
    timestamp: Date.now()
  };
  if (update.message) {
    rememberLog(`[boot] ${update.message}`);
  }
  broadcastBootProgress();
}
async function advanceBootProgress(phase, message, progress) {
  updateBootProgress({
    phase,
    message,
    progress,
    running: true,
    error: null
  });
  if (BOOT_FAKE_MODE) {
    await sleep(BOOT_FAKE_STEP_MS);
  }
}
function fileExists(filePath) {
  try {
    return fs19.statSync(filePath).isFile();
  } catch {
    return false;
  }
}
function directoryExists(filePath) {
  try {
    return fs19.statSync(filePath).isDirectory();
  } catch {
    return false;
  }
}
var UPDATE_WAIT_TIMEOUT_MS = 20 * 60 * 1e3;
var UPDATE_WAIT_POLL_MS = 1e3;
var UPDATE_HANDOFF_DWELL_MS = 2500;
function updateGateDeps() {
  return {
    hasLiveMarker: () => Boolean(readLiveUpdateMarker(HERMES_HOME)),
    isUpdateInFlight: () => updateInFlight
  };
}
async function waitForUpdateToFinish() {
  let announced = false;
  const outcome = await waitForUpdateClearance(updateGateDeps(), {
    onWaitTick: async (reason) => {
      if (!announced) {
        announced = true;
        rememberLog(`[updates] update in progress (${reason}); deferring backend start until it finishes`);
      }
      await advanceBootProgress(
        "backend.update-wait",
        "An update is finishing \u2014 Hermes will start automatically when it completes\u2026",
        12
      );
    },
    pollMs: UPDATE_WAIT_POLL_MS,
    timeoutMs: UPDATE_WAIT_TIMEOUT_MS
  });
  if (outcome === "clear") {
    return false;
  }
  if (outcome === "timeout") {
    rememberLog("[updates] update still in progress after wait timeout; starting backend anyway");
  } else {
    rememberLog("[updates] update finished; proceeding with backend start");
  }
  return true;
}
function unpackedPathFor(filePath) {
  return filePath.replace(/app\.asar(?=$|[\\/])/, "app.asar.unpacked");
}
function findOnPath(command) {
  if (!command) {
    return null;
  }
  if (path22.isAbsolute(command) || command.includes(path22.sep) || IS_WINDOWS3 && command.includes("/")) {
    if (!fileExists(command)) {
      return null;
    }
    if (isWindowsBinaryPathInWsl(command, { isWsl: IS_WSL })) {
      return null;
    }
    return command;
  }
  const pathEntries = String(process.env.PATH || "").split(path22.delimiter).filter(Boolean);
  const extensions = buildPathExtCandidates(process.env.PATHEXT, IS_WINDOWS3);
  for (const entry of pathEntries) {
    for (const extension of extensions) {
      const candidate = path22.join(entry, `${command}${extension}`);
      if (fileExists(candidate)) {
        return candidate;
      }
    }
  }
  return null;
}
function isCommandScript(command) {
  return IS_WINDOWS3 && /\.(cmd|bat)$/i.test(command || "");
}
function unwrapWindowsVenvHermesCommand(command, backendArgs) {
  return resolveVenvHermesCommand(command, backendArgs, {
    isWindows: IS_WINDOWS3,
    isCommandScript,
    fileExists,
    directoryExists,
    canImportHermesCli,
    getVenvPython,
    getVenvSitePackagesEntries,
    buildDesktopBackendEnv,
    hermesHome: HERMES_HOME,
    resolvePath: (...segments) => path22.resolve(...segments),
    dirname: (p2) => path22.dirname(p2),
    basename: (p2) => path22.basename(p2),
    rememberLog
  });
}
var _serveSupportCache = /* @__PURE__ */ new Map();
function backendSupportsServe(backend) {
  if (!backend || !backend.command) {
    return true;
  }
  const key = `${backend.command}::${backend.root || ""}`;
  if (_serveSupportCache.has(key)) {
    return _serveSupportCache.get(key);
  }
  let supported = null;
  if (backend.root) {
    try {
      const src = fs19.readFileSync(path22.join(backend.root, "hermes_cli", "subcommands", "dashboard.py"), "utf8");
      supported = sourceDeclaresServe(src);
    } catch {
      supported = null;
    }
  }
  if (supported === null) {
    try {
      const prefix = backend.args && backend.args[0] === "-m" ? backend.args.slice(0, 2) : [];
      execProbeSync(backend.command, [...prefix, "serve", "--help"], {
        cwd: backend.root || void 0,
        env: { ...process.env, HERMES_HOME, ...backend.env || {} },
        timeout: PROBE_TIMEOUT_MS,
        stdio: "ignore",
        // `.cmd`/`.bat` shim backends carry shell: true in their descriptor
        // (see resolveHermesBackend step 4); execFileSync of a .cmd without
        // shell throws EINVAL on modern Node, which the catch below would
        // mis-cache as "serve unsupported" for the process lifetime.
        shell: Boolean(backend.shell),
        windowsHide: true
      });
      supported = true;
    } catch {
      supported = false;
    }
  }
  _serveSupportCache.set(key, supported);
  rememberLog(
    `[backend] \`serve\` ${supported ? "supported" : "unsupported \u2192 routing via legacy `dashboard`"} for ${backend.label || key}`
  );
  return supported;
}
function getBackendArgsForRuntime(backend) {
  return backendSupportsServe(backend) ? backend.args : dashboardFallbackArgs(backend.args);
}
function normalizeExecutablePathForCompare(commandPath) {
  if (!commandPath) {
    return null;
  }
  let resolved = path22.resolve(String(commandPath));
  try {
    resolved = fs19.realpathSync.native ? fs19.realpathSync.native(resolved) : fs19.realpathSync(resolved);
  } catch {
  }
  return IS_WINDOWS3 ? resolved.toLowerCase() : resolved;
}
function looksLikeDesktopAppBinary(commandPath) {
  if (!IS_WINDOWS3 || !commandPath) {
    return false;
  }
  const normalizedCandidate = normalizeExecutablePathForCompare(commandPath);
  const normalizedCurrentExec = normalizeExecutablePathForCompare(process.execPath);
  if (normalizedCandidate && normalizedCurrentExec && normalizedCandidate === normalizedCurrentExec) {
    return true;
  }
  let resolved = path22.resolve(String(commandPath));
  try {
    resolved = fs19.realpathSync.native ? fs19.realpathSync.native(resolved) : fs19.realpathSync(resolved);
  } catch {
  }
  const resourcesDir = path22.join(path22.dirname(resolved), "resources");
  return fileExists(path22.join(resourcesDir, "app.asar")) || directoryExists(path22.join(resourcesDir, "app.asar.unpacked"));
}
function isHermesSourceRoot(root) {
  return directoryExists(root) && fileExists(path22.join(root, "hermes_cli", "main.py"));
}
function findPythonForRoot(root) {
  const override = process.env.HERMES_DESKTOP_PYTHON;
  if (override && fileExists(override)) {
    return override;
  }
  const relativePaths = IS_WINDOWS3 ? [path22.join(".venv", "Scripts", "python.exe"), path22.join("venv", "Scripts", "python.exe")] : [path22.join(".venv", "bin", "python"), path22.join("venv", "bin", "python")];
  for (const relativePath of relativePaths) {
    const candidate = path22.join(root, relativePath);
    if (fileExists(candidate)) {
      return candidate;
    }
  }
  return findSystemPython();
}
function findSystemPython() {
  if (!IS_WINDOWS3) {
    for (const command of ["python3", "python"]) {
      const candidate = findOnPath(command);
      if (candidate) {
        return candidate;
      }
    }
    return null;
  }
  const SUPPORTED_VERSIONS = ["3.11", "3.12", "3.13"];
  const SUPPORTED_VERSIONS_NO_DOT = ["311", "312", "313"];
  for (const hive of ["HKLM", "HKCU"]) {
    for (const version of SUPPORTED_VERSIONS) {
      try {
        const out = execFileSync6(
          "reg",
          ["query", `${hive}\\SOFTWARE\\Python\\PythonCore\\${version}\\InstallPath`, "/ve", "/reg:64"],
          // Registry reads are near-instant; the bound only exists so a
          // pathologically wedged reg.exe can't hang the synchronous boot
          // resolver forever (this ran unbounded before).
          hiddenWindowsChildOptions({ encoding: "utf8", stdio: ["ignore", "pipe", "ignore"], timeout: 5e3 })
        );
        const match = out.match(/REG_SZ\s+(.+?)\s*$/m);
        if (match) {
          const installPath = match[1].trim();
          const pythonExe = path22.join(installPath, "python.exe");
          if (fileExists(pythonExe)) {
            return pythonExe;
          }
        }
      } catch {
      }
    }
  }
  const programFiles = process.env["ProgramFiles"] || "C:\\Program Files";
  const localAppData = process.env.LOCALAPPDATA || "";
  for (const versionDir of SUPPORTED_VERSIONS_NO_DOT) {
    const systemWide = path22.join(programFiles, `Python${versionDir}`, "python.exe");
    if (fileExists(systemWide)) {
      return systemWide;
    }
    if (localAppData) {
      const perUser = path22.join(localAppData, "Programs", "Python", `Python${versionDir}`, "python.exe");
      if (fileExists(perUser)) {
        return perUser;
      }
    }
  }
  const pyExe = findOnPath("py.exe");
  if (pyExe) {
    for (const version of SUPPORTED_VERSIONS) {
      try {
        const out = execFileSync6(
          pyExe,
          [`-${version}`, "-c", "import sys; print(sys.executable)"],
          hiddenWindowsChildOptions({
            encoding: "utf8",
            stdio: ["ignore", "pipe", "ignore"],
            // Bare interpreter startup — much lighter than the hermes-import
            // probes, but still python.exe under cold cache / AV scan, so
            // share the probe budget rather than running unbounded (this
            // synchronous exec previously had no timeout at all).
            timeout: PROBE_TIMEOUT_MS
          })
        );
        const candidate = out.trim();
        if (candidate && fileExists(candidate)) {
          return candidate;
        }
      } catch {
      }
    }
  }
  return null;
}
function findGitBash2() {
  return findGitBash({
    isWindows: IS_WINDOWS3,
    env: process.env,
    fileExists,
    findOnPath
  });
}
function getVenvPython(venvRoot) {
  return path22.join(venvRoot, IS_WINDOWS3 ? path22.join("Scripts", "python.exe") : path22.join("bin", "python"));
}
function makeDashboardReadyFile() {
  const dir = path22.join(app2.getPath("userData"), "backend-ready");
  fs19.mkdirSync(dir, { recursive: true });
  return path22.join(dir, `dashboard-${process.pid}-${Date.now()}-${crypto6.randomBytes(6).toString("hex")}.json`);
}
var _gitBinaryCache = null;
function resolveGitBinary() {
  if (_gitBinaryCache) {
    return _gitBinaryCache;
  }
  if (!IS_WINDOWS3) {
    _gitBinaryCache = findOnPath("git") || "git";
    return _gitBinaryCache;
  }
  const localAppData = process.env.LOCALAPPDATA || "";
  const candidates = [];
  if (localAppData) {
    candidates.push(path22.join(localAppData, "hermes", "git", "cmd", "git.exe"));
    candidates.push(path22.join(localAppData, "hermes", "git", "bin", "git.exe"));
  }
  candidates.push(path22.join(process.env["ProgramFiles"] || "C:\\Program Files", "Git", "cmd", "git.exe"));
  candidates.push(path22.join(process.env["ProgramFiles(x86)"] || "C:\\Program Files (x86)", "Git", "cmd", "git.exe"));
  if (localAppData) {
    candidates.push(path22.join(localAppData, "Programs", "Git", "cmd", "git.exe"));
  }
  _gitBinaryCache = candidates.find(fileExists) || findOnPath("git") || "git";
  return _gitBinaryCache;
}
var _ghBinaryCache = null;
function resolveGhBinary() {
  if (_ghBinaryCache) {
    return _ghBinaryCache;
  }
  const candidates = [];
  if (IS_WINDOWS3) {
    candidates.push(path22.join(process.env["ProgramFiles"] || "C:\\Program Files", "GitHub CLI", "gh.exe"));
    if (process.env.LOCALAPPDATA) {
      candidates.push(path22.join(process.env.LOCALAPPDATA, "Microsoft", "WinGet", "Links", "gh.exe"));
    }
  } else {
    const home = app2.getPath("home");
    candidates.push("/opt/homebrew/bin/gh", "/usr/local/bin/gh", "/usr/bin/gh", path22.join(home, ".local", "bin", "gh"));
  }
  _ghBinaryCache = candidates.find(fileExists) || findOnPath("gh") || "gh";
  return _ghBinaryCache;
}
function recentHermesLog() {
  return hermesLog.slice(-20).join("\n");
}
function readDesktopUpdateConfig() {
  try {
    const parsed = JSON.parse(fs19.readFileSync(DESKTOP_UPDATE_CONFIG_PATH, "utf8"));
    const branch = typeof parsed?.branch === "string" ? parsed.branch.trim() : "";
    return { branch: branch || DEFAULT_UPDATE_BRANCH };
  } catch {
    return { branch: DEFAULT_UPDATE_BRANCH };
  }
}
function writeFileAtomic(targetPath, data, encoding) {
  const tmp = targetPath + ".tmp";
  fs19.writeFileSync(tmp, data, encoding);
  fs19.renameSync(tmp, targetPath);
}
function writeDesktopUpdateConfig(config) {
  fs19.mkdirSync(path22.dirname(DESKTOP_UPDATE_CONFIG_PATH), { recursive: true });
  writeFileAtomic(DESKTOP_UPDATE_CONFIG_PATH, JSON.stringify(config, null, 2));
}
function readWindowState() {
  try {
    return sanitizeWindowState(JSON.parse(fs19.readFileSync(DESKTOP_WINDOW_STATE_PATH, "utf8")));
  } catch {
    return null;
  }
}
function persistWindowState() {
  if (!mainWindow || mainWindow.isDestroyed() || mainWindow.isMinimized()) {
    return;
  }
  try {
    const { x: x2, y: y2, width, height } = mainWindow.getNormalBounds();
    fs19.mkdirSync(path22.dirname(DESKTOP_WINDOW_STATE_PATH), { recursive: true });
    writeFileAtomic(
      DESKTOP_WINDOW_STATE_PATH,
      JSON.stringify({ x: x2, y: y2, width, height, isMaximized: mainWindow.isMaximized() }, null, 2)
    );
  } catch (err) {
    rememberLog(`[window-state] persist failed: ${err?.message || err}`);
  }
}
var schedulePersistWindowState = debounce(persistWindowState, 250);
var DESKTOP_ZOOM_STATE_PATH = path22.join(app2.getPath("userData"), "zoom-state.json");
function readZoomState() {
  try {
    const raw = JSON.parse(fs19.readFileSync(DESKTOP_ZOOM_STATE_PATH, "utf8"));
    const level = Number(raw?.zoomLevel);
    return Number.isFinite(level) ? level : null;
  } catch {
    return null;
  }
}
function writeZoomState(zoomLevel) {
  try {
    fs19.mkdirSync(path22.dirname(DESKTOP_ZOOM_STATE_PATH), { recursive: true });
    writeFileAtomic(DESKTOP_ZOOM_STATE_PATH, JSON.stringify({ zoomLevel }, null, 2));
  } catch (error) {
    rememberLog(`[zoom] json persist failed: ${error?.message || error}`);
  }
}
function resolveUpdateRoot() {
  const candidates = [
    process.env.HERMES_DESKTOP_HERMES_ROOT && path22.resolve(process.env.HERMES_DESKTOP_HERMES_ROOT),
    !IS_PACKAGED && isHermesSourceRoot(SOURCE_REPO_ROOT) ? SOURCE_REPO_ROOT : null,
    isHermesSourceRoot(ACTIVE_HERMES_ROOT) ? ACTIVE_HERMES_ROOT : null
  ].filter(Boolean);
  return candidates.find((c3) => directoryExists(path22.join(c3, ".git"))) || candidates[0] || ACTIVE_HERMES_ROOT;
}
function runGit2(args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn5(
      resolveGitBinary(),
      IS_WINDOWS3 ? ["-c", "windows.appendAtomically=false", ...args] : args,
      hiddenWindowsChildOptions({
        cwd: options.cwd,
        env: { ...process.env, ...options.env || {}, GIT_TERMINAL_PROMPT: "0" },
        stdio: ["ignore", "pipe", "pipe"]
      })
    );
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      const text = chunk.toString();
      stdout += text;
      options.onLine?.("stdout", text);
    });
    child.stderr.on("data", (chunk) => {
      const text = chunk.toString();
      stderr += text;
      options.onLine?.("stderr", text);
    });
    child.once("error", reject);
    child.once("exit", (code) => resolve({ code, stdout, stderr }));
  });
}
var firstLine = (text) => (text || "").split("\n").find(Boolean) || "";
async function getOriginUrl(updateRoot) {
  const origin = await runGit2(["remote", "get-url", "origin"], { cwd: updateRoot });
  return origin.code === 0 ? origin.stdout.trim() : "";
}
function emitUpdateProgress(payload) {
  const merged = { stage: "idle", message: "", percent: null, error: null, ...payload, at: Date.now() };
  rememberLog(`[updates] ${merged.stage}: ${merged.message || merged.error || ""}`);
  for (const window2 of BrowserWindow2.getAllWindows()) {
    window2.webContents.send("hermes:updates:progress", merged);
  }
}
async function resolveHealedBranch(updateRoot, branch) {
  if (!branch || branch === "main") {
    return branch || "main";
  }
  const originUrl = await getOriginUrl(updateRoot);
  const remote = isOfficialSshRemote(originUrl) ? OFFICIAL_REPO_HTTPS_URL : "origin";
  const probe = await runGit2(["ls-remote", "--exit-code", "--heads", remote, branch], { cwd: updateRoot });
  if (probe.code !== 2) {
    return branch;
  }
  rememberLog(`[updates] origin/${branch} is gone (merged?); falling back to main`);
  const config = readDesktopUpdateConfig();
  if (config.branch !== "main") {
    writeDesktopUpdateConfig({ ...config, branch: "main" });
  }
  return "main";
}
async function checkUpdates() {
  const updateRoot = resolveUpdateRoot();
  let { branch } = readDesktopUpdateConfig();
  const gitDir = path22.join(updateRoot, ".git");
  if (!directoryExists(gitDir)) {
    return {
      supported: false,
      reason: "not-a-git-checkout",
      message: `${updateRoot} isn't a git checkout \u2014 desktop self-update only runs against a source install.`,
      hermesRoot: updateRoot,
      branch
    };
  }
  branch = await resolveHealedBranch(updateRoot, branch);
  const originUrl = await getOriginUrl(updateRoot);
  if (isOfficialSshRemote(originUrl)) {
    const git2 = (args) => runGit2(args, { cwd: updateRoot }).then((r2) => r2.stdout.trim());
    const [currentSha2, target2, dirtyStr2, currentBranch2] = await Promise.all([
      git2(["rev-parse", "HEAD"]),
      runGit2(["ls-remote", OFFICIAL_REPO_HTTPS_URL, `refs/heads/${branch}`], { cwd: updateRoot }),
      git2(["status", "--porcelain"]),
      git2(["rev-parse", "--abbrev-ref", "HEAD"])
    ]);
    const targetSha2 = firstLine(target2.stdout).split(/\s+/)[0] || "";
    if (target2.code !== 0 || !targetSha2) {
      return {
        supported: true,
        branch,
        error: "fetch-failed",
        message: firstLine(target2.stderr) || "git ls-remote failed.",
        hermesRoot: updateRoot,
        fetchedAt: Date.now()
      };
    }
    return {
      supported: true,
      branch,
      currentBranch: currentBranch2,
      behind: currentSha2 && currentSha2 === targetSha2 ? 0 : 1,
      currentSha: currentSha2,
      targetSha: targetSha2,
      commits: [],
      dirty: dirtyStr2.length > 0,
      hermesRoot: updateRoot,
      fetchedAt: Date.now()
    };
  }
  const fetched = await runGit2(["fetch", "--quiet", "origin", branch], { cwd: updateRoot });
  if (fetched.code !== 0) {
    return {
      supported: true,
      branch,
      error: "fetch-failed",
      message: firstLine(fetched.stderr) || "git fetch failed.",
      hermesRoot: updateRoot,
      fetchedAt: Date.now()
    };
  }
  const git = (args) => runGit2(args, { cwd: updateRoot }).then((r2) => r2.stdout.trim());
  const [currentSha, targetSha, dirtyStr, currentBranch, shallowStr, mergeBaseStr] = await Promise.all([
    git(["rev-parse", "HEAD"]),
    git(["rev-parse", `origin/${branch}`]),
    git(["status", "--porcelain"]),
    git(["rev-parse", "--abbrev-ref", "HEAD"]),
    git(["rev-parse", "--is-shallow-repository"]),
    // merge-base exits non-zero with empty stdout when HEAD shares no common
    // ancestor with the freshly fetched tip — exactly the shallow-clone case.
    git(["merge-base", "HEAD", `origin/${branch}`])
  ]);
  const isShallow = shallowStr === "true";
  const hasMergeBase = Boolean(mergeBaseStr);
  const countStr = shouldCountCommits({ isShallow, hasMergeBase }) ? await git(["rev-list", `HEAD..origin/${branch}`, "--count"]) : "";
  const behind = resolveBehindCount({
    countStr,
    currentSha,
    targetSha,
    isShallow,
    hasMergeBase
  });
  const commits = behind > 0 ? await readCommitLog(updateRoot, branch) : [];
  return {
    supported: true,
    branch,
    currentBranch,
    behind,
    currentSha,
    targetSha,
    commits,
    dirty: dirtyStr.length > 0,
    hermesRoot: updateRoot,
    fetchedAt: Date.now()
  };
}
async function readCommitLog(cwd, branch) {
  const SEP = "";
  const REC = "";
  const { stdout } = await runGit2(
    ["log", `HEAD..origin/${branch}`, `--pretty=format:%H${SEP}%s${SEP}%an${SEP}%at${REC}`, "-n", "40"],
    { cwd }
  );
  return stdout.split(REC).map((line) => line.trim()).filter(Boolean).map((line) => {
    const [sha, summary, author, at] = line.split(SEP);
    return { sha, summary, author, at: Number.parseInt(at, 10) * 1e3 };
  });
}
var updateInFlight = false;
var isQuittingForHandoff = false;
var isReallyQuitting = false;
var quitPromptOpen = false;
var quitConfirmedWithActiveWork = false;
function resolveUpdaterBinary() {
  return resolveStagedUpdaterBinary(HERMES_HOME, { fileExists, isWindows: IS_WINDOWS3 });
}
function repairMacUpdaterHelper(updater) {
  if (!IS_MAC || !updater) {
    return;
  }
  try {
    execFileSync6("/usr/bin/xattr", ["-cr", updater], { stdio: "ignore" });
  } catch (err) {
    rememberLog(`[updates] macOS updater helper quarantine repair skipped: ${err.message}`);
  }
  try {
    execFileSync6("/usr/bin/codesign", ["--verify", updater], { stdio: "ignore" });
    return;
  } catch {
  }
  try {
    execFileSync6("/usr/bin/codesign", ["--force", "--sign", "-", updater], { stdio: "ignore" });
    rememberLog("[updates] repaired macOS updater helper signature");
  } catch (err) {
    rememberLog(`[updates] macOS updater helper signature repair skipped: ${err.message}`);
  }
}
function venvHermesShimPath(updateRoot) {
  return IS_WINDOWS3 ? path22.join(updateRoot, "venv", "Scripts", "hermes.exe") : path22.join(updateRoot, "venv", "bin", "hermes");
}
function isShimLocked(shimPath) {
  if (!IS_WINDOWS3) {
    return false;
  }
  let fd;
  try {
    fd = fs19.openSync(shimPath, "r+");
    return false;
  } catch (err) {
    return err && err.code !== "ENOENT";
  } finally {
    if (fd !== void 0) {
      try {
        fs19.closeSync(fd);
      } catch {
      }
    }
  }
}
function forceKillProcessTree(pid) {
  if (!IS_WINDOWS3) {
    return;
  }
  if (!Number.isInteger(pid) || pid <= 0) {
    return;
  }
  try {
    execFileSync6("taskkill", ["/PID", String(pid), "/T", "/F"], hiddenWindowsChildOptions({ stdio: "ignore" }));
  } catch {
  }
}
async function releaseBackendLockForUpdate(updateRoot) {
  return releaseBackendLock(updateRoot, "updates");
}
async function releaseBackendLock(updateRoot, tag) {
  if (!IS_WINDOWS3) {
    return { unlocked: true };
  }
  const pids = [];
  const hermesProcess = backendConnectionState.getProcess();
  if (hermesProcess && Number.isInteger(hermesProcess.pid)) {
    pids.push(hermesProcess.pid);
  }
  for (const entry of backendPool.values()) {
    if (entry.process && Number.isInteger(entry.process.pid)) {
      pids.push(entry.process.pid);
    }
  }
  if (hermesProcess && !hermesProcess.killed) {
    try {
      hermesProcess.kill("SIGTERM");
    } catch {
    }
  }
  stopAllPoolBackends();
  for (const pid of pids) {
    forceKillProcessTree(pid);
  }
  const shim = venvHermesShimPath(updateRoot);
  const deadlineMs = Date.now() + 15e3;
  while (Date.now() < deadlineMs) {
    if (!isShimLocked(shim)) {
      rememberLog(`[${tag}] venv shim unlocked; safe to proceed`);
      return { unlocked: true };
    }
    const stragglers = [];
    const currentHermesProcess = backendConnectionState.getProcess();
    if (currentHermesProcess && Number.isInteger(currentHermesProcess.pid)) {
      stragglers.push(currentHermesProcess.pid);
    }
    for (const entry of backendPool.values()) {
      if (entry.process && Number.isInteger(entry.process.pid)) {
        stragglers.push(entry.process.pid);
      }
    }
    for (const pid of stragglers) {
      forceKillProcessTree(pid);
    }
    await new Promise((r2) => setTimeout(r2, 300));
  }
  rememberLog(
    `[${tag}] venv shim still locked after 15s; aborting hand-off (something outside this app holds the venv)`
  );
  return { unlocked: false };
}
async function applyUpdates(opts = {}) {
  if (updateInFlight) {
    throw new Error("An update is already in progress.");
  }
  updateInFlight = true;
  try {
    const updater = resolveUpdaterBinary();
    if (!updater && !IS_WINDOWS3) {
      return await applyUpdatesPosixInApp(opts);
    }
    if (!updater) {
      const updateRoot2 = resolveUpdateRoot();
      let command = "hermes update";
      try {
        const head = await runGit2(["rev-parse", "--abbrev-ref", "HEAD"], { cwd: updateRoot2 });
        const current = (head.stdout || "").trim();
        if (head.code === 0 && current && current !== "HEAD") {
          const branch2 = await resolveHealedBranch(updateRoot2, current);
          if (branch2 !== "main") {
            command = `hermes update --branch ${branch2}`;
          }
        }
      } catch {
      }
      rememberLog(`[updates] no staged updater; surfacing manual \`${command}\` for CLI install at ${updateRoot2}`);
      emitUpdateProgress({ stage: "manual", message: command, percent: null });
      return { ok: true, manual: true, command, hermesRoot: updateRoot2 };
    }
    const handoffConflict = updateHandoffConflict(HERMES_HOME);
    if (handoffConflict) {
      rememberLog(`[updates] refusing hand-off: ${handoffConflict.message}`);
      emitUpdateProgress({ stage: "error", message: handoffConflict.message, percent: null });
      return { ok: false, error: "update-already-running", message: handoffConflict.message };
    }
    emitUpdateProgress({
      stage: "restart",
      message: "Updating Hermes \u2014 this window will close and the updater will open. Don\u2019t reopen Hermes yourself; it restarts automatically when the update finishes.",
      percent: 100
    });
    repairMacUpdaterHelper(updater);
    const updateRoot = resolveUpdateRoot();
    const { branch: configuredBranch } = readDesktopUpdateConfig();
    const branch = await resolveHealedBranch(updateRoot, configuredBranch || DEFAULT_UPDATE_BRANCH);
    const updaterArgs = ["--update", "--branch", branch];
    const targetApp = IS_MAC ? runningAppBundle() : null;
    if (targetApp) {
      updaterArgs.push("--target-app", targetApp);
    }
    const venvBin = path22.join(updateRoot, "venv", IS_WINDOWS3 ? "Scripts" : "bin");
    preflightStateDb(HERMES_HOME, rememberLog);
    const lock = await releaseBackendLockForUpdate(updateRoot);
    if (!lock.unlocked) {
      const message = "Update aborted: another process is holding the Hermes install open (a second Hermes window or a terminal running hermes?). Close it and retry.";
      emitUpdateProgress({ stage: "error", message, percent: null });
      startHermes().catch(() => {
      });
      return { ok: false, error: message };
    }
    if (IS_WINDOWS3) {
      const scanOutcome = await scanVenvBlockers(updateRoot);
      if (scanOutcome.kind === "blocked") {
        const message = formatBlockerMessage(scanOutcome.result);
        rememberLog(`[updates] venv-blocked: ${scanOutcome.result.processes.length} process(es) hold the install`);
        emitUpdateProgress({ stage: "error", message, percent: null });
        startHermes().catch(() => {
        });
        return { ok: false, error: "venv-blocked", message };
      }
      if (scanOutcome.kind === "probe-failure") {
        const message = formatProbeFailedMessage();
        rememberLog(`[updates] venv-blocker probe failed: ${scanOutcome.error}`);
        emitUpdateProgress({ stage: "error", message, percent: null });
        startHermes().catch(() => {
        });
        return { ok: false, error: "venv-probe-failed", message };
      }
    }
    const child = spawnUpdaterProcess(updater, updaterArgs, {
      cwd: HERMES_HOME,
      env: {
        ...process.env,
        HERMES_HOME,
        PATH: pathWithHermesManagedNode(venvBin)
      },
      detached: true,
      stdio: "ignore"
    });
    if (Number.isInteger(child.pid) && stagedUpdaterSupportsPrewrittenMarker(updater)) {
      writeUpdateMarker(HERMES_HOME, child.pid);
    } else if (Number.isInteger(child.pid)) {
      rememberLog(
        `[updates] skipping marker pre-write: staged updater predates self-adopt (${updater}); it would refuse its own claim`
      );
    }
    rememberLog(`[updates] launched updater: ${updater} ${updaterArgs.join(" ")}; exiting desktop to release venv shim`);
    isQuittingForHandoff = true;
    setTimeout(() => {
      app2.quit();
    }, UPDATE_HANDOFF_DWELL_MS);
    return { ok: true, handedOff: true, updater };
  } finally {
    updateInFlight = false;
  }
}
async function handOffWindowsBootstrapRecovery(reason) {
  if (!IS_WINDOWS3 || !IS_PACKAGED) {
    return false;
  }
  const updater = resolveUpdaterBinary();
  if (!updater) {
    return false;
  }
  const handoffConflict = updateHandoffConflict(HERMES_HOME);
  if (handoffConflict) {
    rememberLog(`[bootstrap] refusing recovery hand-off: ${handoffConflict.message}`);
    isQuittingForHandoff = true;
    setTimeout(() => {
      app2.quit();
    }, UPDATE_HANDOFF_DWELL_MS);
    return true;
  }
  const updateRoot = resolveUpdateRoot();
  const { branch: configuredBranch } = readDesktopUpdateConfig();
  const branch = directoryExists(path22.join(updateRoot, ".git")) ? await resolveHealedBranch(updateRoot, configuredBranch || DEFAULT_UPDATE_BRANCH) : configuredBranch || DEFAULT_UPDATE_BRANCH;
  const venvBin = path22.join(updateRoot, "venv", IS_WINDOWS3 ? "Scripts" : "bin");
  const venvHermes = path22.join(venvBin, IS_WINDOWS3 ? "hermes.exe" : "hermes");
  const venvPython = path22.join(venvBin, IS_WINDOWS3 ? "python.exe" : "python");
  const haveRealInstall = fileExists(venvPython) || fileExists(venvHermes) || fileExists(path22.join(updateRoot, ".hermes-bootstrap-complete"));
  const updaterArgs = chooseUpdaterArgs(haveRealInstall, branch);
  await releaseBackendLockForUpdate(updateRoot);
  const child = spawnUpdaterProcess(updater, updaterArgs, {
    cwd: HERMES_HOME,
    env: {
      ...process.env,
      HERMES_HOME,
      PATH: pathWithHermesManagedNode(venvBin)
    },
    detached: true,
    stdio: "ignore"
  });
  if (Number.isInteger(child.pid) && stagedUpdaterSupportsPrewrittenMarker(updater)) {
    writeUpdateMarker(HERMES_HOME, child.pid);
  } else if (Number.isInteger(child.pid)) {
    rememberLog(
      `[bootstrap] skipping marker pre-write: staged updater predates self-adopt (${updater}); it would refuse its own claim`
    );
  }
  rememberLog(
    `[bootstrap] handed off ${reason} recovery to updater: ${updater} ${updaterArgs.join(" ")}; exiting desktop to release app.asar`
  );
  isQuittingForHandoff = true;
  setTimeout(() => {
    app2.quit();
  }, UPDATE_HANDOFF_DWELL_MS);
  return true;
}
function resolveHermesCliBinary(updateRoot) {
  const venvHermes = path22.join(updateRoot, "venv", "bin", "hermes");
  if (fileExists(venvHermes)) {
    return venvHermes;
  }
  return findOnPath("hermes") || null;
}
function runStreamedUpdate(command, args, { cwd, env: env2, stage } = {}) {
  return new Promise((resolve) => {
    let child;
    try {
      child = spawn5(
        command,
        args,
        hiddenWindowsChildOptions({
          cwd,
          env: { ...process.env, ...env2 || {} },
          stdio: ["ignore", "pipe", "pipe"]
        })
      );
    } catch (err) {
      resolve({ code: 1, error: err.message });
      return;
    }
    const emitLines = (chunk) => {
      for (const line of chunk.toString().split("\n")) {
        const trimmed2 = line.trim();
        if (trimmed2) {
          emitUpdateProgress({ stage, message: trimmed2, percent: null });
        }
      }
    };
    child.stdout.on("data", emitLines);
    child.stderr.on("data", emitLines);
    child.once("error", (err) => resolve({ code: 1, error: err.message }));
    child.once("exit", (code) => resolve({ code }));
  });
}
function runningAppBundle() {
  if (!IS_MAC) {
    return null;
  }
  let dir = path22.dirname(app2.getPath("exe"));
  for (let i2 = 0; i2 < 2; i2++) {
    dir = path22.dirname(dir);
  }
  return dir.endsWith(".app") ? dir : null;
}
function preflightStateDb(hermesHome, rememberLog2) {
  const stateDbPath = path22.join(hermesHome, "state.db");
  if (!fileExists(stateDbPath)) {
    rememberLog2("[updates] state.db pre-flight: not found (fresh install?)");
    return;
  }
  try {
    const stat = fs19.statSync(stateDbPath);
    if (stat.size > 100) {
      const fd = fs19.openSync(stateDbPath, "r");
      const header = Buffer.alloc(16);
      fs19.readSync(fd, header, 0, 16, 0);
      fs19.closeSync(fd);
      const expectedHeader = Buffer.from("SQLite format 3\0");
      const headerOk = header.equals(expectedHeader);
      rememberLog2(
        `[updates] state.db pre-flight: size=${stat.size}, headerOk=${headerOk}, headerHex=${header.toString("hex")}`
      );
      if (!headerOk) {
        rememberLog2(
          "[updates] state.db header is INVALID before update \u2014 this indicates pre-existing corruption or a concurrent write issue"
        );
      }
      const ts = (/* @__PURE__ */ new Date()).toISOString().replace(/[:.]/g, "-");
      const emergencyPath = path22.join(hermesHome, `state.db.pre-update-emergency-${ts}.bak`);
      try {
        fs19.copyFileSync(stateDbPath, emergencyPath);
        const emergStat = fs19.statSync(emergencyPath);
        rememberLog2(`[updates] emergency state.db backup: ${emergencyPath} (${emergStat.size} bytes)`);
        try {
          const homeDir = fs19.readdirSync(hermesHome);
          const backups = homeDir.filter(
            (f) => f.startsWith("state.db.pre-update-emergency-") && f.endsWith(".bak") && f !== path22.basename(emergencyPath)
          ).sort().reverse();
          for (const old of backups.slice(2)) {
            try {
              fs19.unlinkSync(path22.join(hermesHome, old));
            } catch {
            }
          }
        } catch {
        }
      } catch (copyErr) {
        rememberLog2(`[updates] emergency state.db backup failed: ${copyErr.message}`);
      }
    } else {
      rememberLog2(`[updates] state.db too small (${stat.size} bytes) for a valid SQLite database`);
    }
  } catch (statErr) {
    rememberLog2(`[updates] could not stat state.db before update: ${statErr.message}`);
  }
}
function shellQuote2(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}
async function applyUpdatesPosixInApp(opts) {
  const updateRoot = resolveUpdateRoot();
  const hermes = resolveHermesCliBinary(updateRoot);
  if (!hermes) {
    emitUpdateProgress({ stage: "manual", message: "hermes update", percent: null });
    return { ok: true, manual: true, command: "hermes update", hermesRoot: updateRoot };
  }
  preflightStateDb(HERMES_HOME, rememberLog);
  const env2 = {
    HERMES_HOME,
    PYTHONUNBUFFERED: "1",
    PATH: pathWithHermesManagedNode(path22.join(updateRoot, "venv", "bin"))
  };
  const desktopChildPids = [];
  const hermesProcess = backendConnectionState.getProcess();
  if (hermesProcess && Number.isInteger(hermesProcess.pid)) {
    desktopChildPids.push(hermesProcess.pid);
  }
  for (const entry of backendPool.values()) {
    if (entry.process && Number.isInteger(entry.process.pid)) {
      desktopChildPids.push(entry.process.pid);
    }
  }
  if (desktopChildPids.length) {
    env2.HERMES_DESKTOP_CHILD_PID = desktopChildPids.join(",");
  }
  let branchArgs = [];
  try {
    const head = await runGit2(["rev-parse", "--abbrev-ref", "HEAD"], { cwd: updateRoot });
    const current = (head.stdout || "").trim();
    if (head.code === 0 && current && current !== "HEAD") {
      branchArgs = ["--branch", await resolveHealedBranch(updateRoot, current)];
    }
  } catch {
  }
  emitUpdateProgress({ stage: "update", message: "Updating Hermes (git + dependencies)\u2026", percent: 10 });
  const updated = await runStreamedUpdate(hermes, ["update", "--yes", ...branchArgs], {
    cwd: updateRoot,
    env: env2,
    stage: "update"
  });
  if (updated.code !== 0) {
    emitUpdateProgress({ stage: "error", message: "hermes update failed.", error: updated.error || "update-failed" });
    return { ok: false, error: "hermes update failed" };
  }
  emitUpdateProgress({ stage: "rebuild", message: "Rebuilding the desktop app\u2026", percent: 60 });
  const rebuilt = await runRebuildWithRetry((attempt) => {
    if (attempt > 0) {
      emitUpdateProgress({ stage: "rebuild", message: "Retrying the desktop rebuild\u2026", percent: 60 });
    }
    return runStreamedUpdate(hermes, ["desktop", "--build-only"], { cwd: updateRoot, env: env2, stage: "rebuild" });
  });
  if (rebuilt.code !== 0) {
    emitUpdateProgress({
      stage: "error",
      message: "Backend updated, but the desktop rebuild failed. Restart Hermes to retry.",
      error: rebuilt.error || "rebuild-failed"
    });
    return { ok: false, backendUpdated: true, error: "desktop rebuild failed" };
  }
  if (!IS_MAC) {
    const unpackedDir = resolveUnpackedRelease(process.execPath, updateRoot, process.platform);
    const underUnpacked = unpackedDir !== null;
    const preflight = underUnpacked ? sandboxPreflight(unpackedDir, (p2) => fs19.statSync(p2)) : { ok: false, reason: "not-under-unpacked", path: null };
    const sandboxFallback = sandboxFallbackFromEnv(process.env, process.argv.slice(1));
    const sandboxOk = preflight.ok || sandboxFallback;
    if (underUnpacked && !preflight.ok) {
      rememberLog(
        `[updates] sandbox preflight: not launchable (${preflight.reason}) at ${preflight.path}; fallback=${sandboxFallback ? "env/--no-sandbox" : "none"}`
      );
    }
    const outcome = decideRelaunchOutcome({ underUnpacked, sandboxOk });
    if (outcome === "relaunch") {
      emitUpdateProgress({ stage: "restart", message: "Restarting Hermes\u2026", percent: 100 });
      const relaunchArgs = collectRelaunchArgs(process.argv.slice(1));
      const relaunchEnv = collectRelaunchEnv(process.env);
      const relaunchScript = buildRelaunchScript({
        pid: process.pid,
        execPath: process.execPath,
        args: relaunchArgs,
        env: relaunchEnv,
        cwd: process.cwd()
      });
      const scriptPath2 = path22.join(app2.getPath("temp"), `hermes-desktop-update-${Date.now()}.sh`);
      try {
        fs19.writeFileSync(scriptPath2, relaunchScript, { mode: 493 });
        const child2 = spawn5("/bin/bash", [scriptPath2], { detached: true, stdio: "ignore" });
        child2.unref();
        rememberLog(
          `[updates] launched linux relaunch: ${scriptPath2} -> ${process.execPath} (args=${relaunchArgs.length}, env=${Object.keys(relaunchEnv).length})`
        );
        isQuittingForHandoff = true;
        setTimeout(() => app2.quit(), UPDATE_HANDOFF_DWELL_MS);
        return { ok: true, handedOff: true };
      } catch (err) {
        rememberLog(`[updates] linux relaunch failed: ${err.message}; falling back to manual restart`);
        return {
          ok: true,
          backendUpdated: true,
          guiUpdated: false,
          manualRestart: true,
          message: "Backend updated. Quit and reopen Hermes to load the new version."
        };
      }
    }
    if (outcome === "guiSkew") {
      emitUpdateProgress({
        stage: "guiSkew",
        message: "Backend updated, but the desktop app package was not changed. Update or reinstall the Hermes desktop app to match.",
        percent: 100
      });
      rememberLog(
        `[updates] gui/backend skew: execPath ${process.execPath} not under release/*-unpacked; backend updated, GUI package unchanged (AppImage/.deb/.rpm/dev/unresolved)`
      );
      return { ok: true, backendUpdated: true, guiUpdated: false, guiSkew: true };
    }
    rememberLog(
      `[updates] sandbox not launchable (${preflight.reason}); skipping auto-relaunch, returning manual-restart so the user keeps a working window`
    );
    return {
      ok: true,
      backendUpdated: true,
      guiUpdated: false,
      manualRestart: true,
      sandboxBlocked: true,
      message: "Backend updated. The rebuilt app can\u2019t relaunch automatically (sandbox helper needs root). Quit and reopen Hermes to finish."
    };
  }
  const rebuiltApp = [
    path22.join(updateRoot, "apps", "desktop", "release", "mac-arm64", "Hermes.app"),
    path22.join(updateRoot, "apps", "desktop", "release", "mac", "Hermes.app")
  ].find(directoryExists);
  const targetApp = runningAppBundle();
  if (!rebuiltApp || !targetApp) {
    emitUpdateProgress({
      stage: "done",
      message: "Backend updated. Restart Hermes to load the new version.",
      percent: 100
    });
    return { ok: true, backendUpdated: true, rebuiltApp: rebuiltApp || null };
  }
  emitUpdateProgress({ stage: "restart", message: "Installing the updated app and restarting\u2026", percent: 95 });
  const swapScript = `#!/bin/bash
set -u
APP_PID=${process.pid}
SRC=${shellQuote2(rebuiltApp)}
DST=${shellQuote2(targetApp)}
for _ in $(seq 1 240); do
  kill -0 "$APP_PID" 2>/dev/null || break
  sleep 0.5
done
if [ "$SRC" != "$DST" ]; then
  if /usr/bin/ditto "$SRC" "$DST.hermes-update-new"; then
    rm -rf "$DST.hermes-update-old" 2>/dev/null || true
    mv "$DST" "$DST.hermes-update-old" 2>/dev/null || rm -rf "$DST"
    mv "$DST.hermes-update-new" "$DST"
    rm -rf "$DST.hermes-update-old" 2>/dev/null || true
  fi
fi
/usr/bin/xattr -dr com.apple.quarantine "$DST" 2>/dev/null || true
/usr/bin/open "$DST"
`;
  const scriptPath = path22.join(app2.getPath("temp"), `hermes-desktop-update-${Date.now()}.sh`);
  try {
    fs19.writeFileSync(scriptPath, swapScript, { mode: 493 });
  } catch (err) {
    emitUpdateProgress({
      stage: "done",
      message: "Backend + app updated. Restart Hermes to load the new version.",
      percent: 100
    });
    rememberLog(`[updates] could not write swap script: ${err.message}; rebuilt app at ${rebuiltApp}`);
    return { ok: true, backendUpdated: true, rebuiltApp };
  }
  const child = spawn5("/bin/bash", [scriptPath], { detached: true, stdio: "ignore" });
  child.unref();
  rememberLog(`[updates] launched mac swap+relaunch: ${scriptPath} (${rebuiltApp} -> ${targetApp})`);
  isQuittingForHandoff = true;
  setTimeout(() => app2.quit(), 600);
  return { ok: true, handedOff: true, rebuiltApp, targetApp };
}
function readJson(filePath) {
  try {
    return JSON.parse(fs19.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}
function readBootstrapMarker() {
  return readJson(BOOTSTRAP_COMPLETE_MARKER);
}
function isActiveRuntimeUsable() {
  const venvPython = getVenvPython(VENV_ROOT);
  return isHermesSourceRoot(ACTIVE_HERMES_ROOT) && fileExists(venvPython) && canImportHermesCli(venvPython, {
    env: {
      PYTHONPATH: [ACTIVE_HERMES_ROOT, process.env.PYTHONPATH].filter(Boolean).join(path22.delimiter)
    }
  });
}
function activeRuntimeState() {
  return classifyActiveRuntime(readBootstrapMarker(), BOOTSTRAP_MARKER_SCHEMA_VERSION, isActiveRuntimeUsable());
}
function writeBootstrapMarker(payload) {
  fs19.mkdirSync(path22.dirname(BOOTSTRAP_COMPLETE_MARKER), { recursive: true });
  const merged = {
    schemaVersion: BOOTSTRAP_MARKER_SCHEMA_VERSION,
    pinnedCommit: payload.pinnedCommit || null,
    pinnedBranch: payload.pinnedBranch || null,
    completedAt: (/* @__PURE__ */ new Date()).toISOString(),
    desktopVersion: app2.getVersion()
  };
  writeFileAtomic(BOOTSTRAP_COMPLETE_MARKER, JSON.stringify(merged, null, 2) + "\n", "utf8");
  return merged;
}
function resolveWebDist() {
  const override = process.env.HERMES_DESKTOP_WEB_DIST;
  if (override && directoryExists(path22.resolve(override))) {
    return path22.resolve(override);
  }
  const unpackedDist = path22.join(unpackedPathFor(APP_ROOT), "dist");
  if (directoryExists(unpackedDist)) {
    return unpackedDist;
  }
  const fallback = path22.join(APP_ROOT, "dist");
  if (IS_PACKAGED && /app\.asar(?=$|[\\/])/.test(fallback) && !directoryExists(fallback)) {
    rememberLog(
      `[web-dist] dashboard frontend dir resolved to an asar-internal path that is not a real directory: ${fallback}. Static routes will 404. Ensure dist/** is unpacked (asarUnpack) or set HERMES_DESKTOP_WEB_DIST.`
    );
  }
  return fallback;
}
function resolveRendererIndex() {
  const candidates = [path22.join(APP_ROOT, "dist", "index.html"), path22.join(resolveWebDist(), "index.html")];
  const found = candidates.find(fileExists);
  if (found) {
    return found;
  }
  rememberLog(
    `[renderer] index.html not found \u2014 the desktop app was packaged without a renderer bundle. Tried: ${candidates.join(", ")}. Rebuild with: hermes desktop --force-build`
  );
  return candidates[0];
}
function isPackagedInstallPath2(dir) {
  return isPackagedInstallPath(dir, {
    isPackaged: IS_PACKAGED,
    installRoots: [
      APP_ROOT,
      path22.dirname(process.execPath),
      resolveRemovableAppPath(process.execPath, process.platform, process.env)
    ]
  });
}
function resolveHermesCwd() {
  const candidates = [
    readDefaultProjectDir(),
    process.env.HERMES_DESKTOP_CWD,
    IS_PACKAGED ? null : process.env.INIT_CWD,
    IS_PACKAGED ? null : process.cwd(),
    !IS_PACKAGED ? SOURCE_REPO_ROOT : null,
    app2.getPath("home")
  ];
  for (const candidate of candidates) {
    if (!candidate) {
      continue;
    }
    const resolved = path22.resolve(String(candidate));
    if (isPackagedInstallPath2(resolved)) {
      continue;
    }
    if (directoryExists(resolved)) {
      return resolved;
    }
  }
  return app2.getPath("home");
}
function sanitizeWorkspaceCwd(cwd) {
  const trimmed2 = typeof cwd === "string" ? cwd.trim() : "";
  if (!trimmed2 || isPackagedInstallPath2(trimmed2)) {
    return { cwd: resolveHermesCwd(), sanitized: Boolean(trimmed2) };
  }
  try {
    const resolved = path22.resolve(trimmed2);
    if (directoryExists(resolved)) {
      return { cwd: resolved, sanitized: false };
    }
  } catch {
  }
  return { cwd: resolveHermesCwd(), sanitized: Boolean(trimmed2) };
}
var DEFAULT_PROJECT_DIR_CONFIG_FILENAME = "project-dir.json";
function defaultProjectDirConfigPath() {
  return path22.join(app2.getPath("userData"), DEFAULT_PROJECT_DIR_CONFIG_FILENAME);
}
function readDefaultProjectDir() {
  try {
    const raw = fs19.readFileSync(defaultProjectDirConfigPath(), "utf8");
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed.dir === "string" && parsed.dir.trim()) {
      const resolved = path22.resolve(parsed.dir);
      if (directoryExists(resolved)) {
        return resolved;
      }
    }
  } catch {
  }
  return null;
}
function writeDefaultProjectDir(dir) {
  const target2 = defaultProjectDirConfigPath();
  const payload = dir ? JSON.stringify({ dir: path22.resolve(dir) }, null, 2) : JSON.stringify({}, null, 2);
  try {
    fs19.mkdirSync(path22.dirname(target2), { recursive: true });
    fs19.writeFileSync(target2, payload, "utf8");
  } catch (error) {
    rememberLog(`[settings] write default project dir failed: ${error.message}`);
  }
}
function createPythonBackend(root, label, backendArgs, options = {}) {
  const python = findPythonForRoot(root);
  if (!python) {
    return null;
  }
  const venvRoot = path22.join(root, "venv");
  const venvPython = getVenvPython(venvRoot);
  const command = IS_WINDOWS3 && fileExists(venvPython) ? venvPython : python;
  return {
    kind: "python",
    label,
    command,
    args: ["-m", "hermes_cli.main", ...backendArgs],
    env: buildDesktopBackendEnv({
      hermesHome: HERMES_HOME,
      pythonPathEntries: [root, ...getVenvSitePackagesEntries(venvRoot)],
      venvRoot
    }),
    root,
    bootstrap: Boolean(options.bootstrap),
    shell: false
  };
}
function createActiveBackend(backendArgs) {
  const venvPython = getVenvPython(VENV_ROOT);
  const command = fileExists(venvPython) ? venvPython : findSystemPython();
  return {
    kind: "python",
    label: `Hermes at ${ACTIVE_HERMES_ROOT}`,
    command,
    args: ["-m", "hermes_cli.main", ...backendArgs],
    env: buildDesktopBackendEnv({
      hermesHome: HERMES_HOME,
      pythonPathEntries: [ACTIVE_HERMES_ROOT, ...getVenvSitePackagesEntries(VENV_ROOT)],
      venvRoot: VENV_ROOT
    }),
    root: ACTIVE_HERMES_ROOT,
    bootstrap: true,
    shell: false
  };
}
function resolveHermesBackend(backendArgs) {
  const overrideRoot = process.env.HERMES_DESKTOP_HERMES_ROOT && path22.resolve(process.env.HERMES_DESKTOP_HERMES_ROOT);
  if (overrideRoot && isHermesSourceRoot(overrideRoot)) {
    const backend = createPythonBackend(overrideRoot, `Hermes source at ${overrideRoot}`, backendArgs);
    if (backend) {
      return backend;
    }
  }
  if (!IS_PACKAGED && isHermesSourceRoot(SOURCE_REPO_ROOT)) {
    const backend = createPythonBackend(SOURCE_REPO_ROOT, `Hermes source at ${SOURCE_REPO_ROOT}`, backendArgs);
    if (backend) {
      return backend;
    }
  }
  const activeRuntime = activeRuntimeState();
  if (activeRuntime.shouldUseActiveRuntime && !bootstrapRepairRequested) {
    if (!activeRuntime.hasValidMarker) {
      rememberLog(
        `[bootstrap] Active Hermes runtime at ${ACTIVE_HERMES_ROOT} is usable but the bootstrap marker is missing or stale; skipping first-run bootstrap.`
      );
    }
    return createActiveBackend(backendArgs);
  }
  if (bootstrapRepairRequested) {
    rememberLog("[bootstrap] repair requested; bypassing the usable active runtime to re-run the installer");
  }
  if (process.env.HERMES_DESKTOP_IGNORE_EXISTING !== "1") {
    let hermesCommand = null;
    const hermesOverride = process.env.HERMES_DESKTOP_HERMES;
    if (hermesOverride) {
      const resolvedOverride = findOnPath(hermesOverride);
      if (resolvedOverride) {
        hermesCommand = resolvedOverride;
      } else if (!isWindowsBinaryPathInWsl(hermesOverride, { isWsl: IS_WSL })) {
        hermesCommand = hermesOverride;
      } else {
        rememberLog(`Ignoring Windows Hermes override under WSL: ${hermesOverride}`);
      }
    } else {
      hermesCommand = findOnPath("hermes");
    }
    if (hermesCommand) {
      if (looksLikeDesktopAppBinary(hermesCommand)) {
        rememberLog(`Ignoring desktop app executable on PATH while resolving Hermes CLI: ${hermesCommand}`);
        hermesCommand = null;
      }
    }
    if (hermesCommand) {
      const unwrapped = unwrapWindowsVenvHermesCommand(hermesCommand, backendArgs);
      if (unwrapped) {
        return unwrapped;
      }
      const shellForProbe = isCommandScript(hermesCommand);
      if (shouldTrustHermesOverride(hermesOverride) || verifyHermesCli(hermesCommand, { shell: shellForProbe })) {
        return {
          label: `existing Hermes CLI at ${hermesCommand}`,
          command: hermesCommand,
          args: backendArgs,
          bootstrap: false,
          env: {},
          kind: "command",
          shell: shellForProbe
        };
      }
      rememberLog(
        `Ignoring existing Hermes CLI at ${hermesCommand}: --version probe failed; falling through to bootstrap.`
      );
    }
  }
  const python = findSystemPython();
  if (python) {
    if (canImportHermesCli(python)) {
      return {
        kind: "python",
        label: `installed hermes_cli module via ${python}`,
        command: python,
        args: ["-m", "hermes_cli.main", ...backendArgs],
        bootstrap: false,
        env: {},
        shell: false
      };
    }
    rememberLog(`Ignoring system Python ${python}: hermes_cli is not importable; falling through to bootstrap.`);
  }
  return {
    kind: "bootstrap-needed",
    label: "Hermes Agent not installed yet; bootstrap required",
    command: null,
    args: backendArgs,
    bootstrap: true,
    env: {},
    shell: false,
    // Hints for the bootstrap runner / UI layer:
    activeRoot: ACTIVE_HERMES_ROOT,
    installStamp: INSTALL_STAMP,
    // may be null in dev
    isPackaged: IS_PACKAGED,
    platform: process.platform
  };
}
async function ensureRuntime(backend) {
  if (!backend.bootstrap) {
    await advanceBootProgress("runtime.external", `Using ${backend.label}`, 32);
    return backend;
  }
  if (backend.kind === "bootstrap-needed") {
    rememberLog("[bootstrap] no Hermes install found; starting first-launch bootstrap");
    if (await handOffWindowsBootstrapRecovery("bootstrap-needed")) {
      const handoffError = new Error(
        "Hermes recovery was handed off to Hermes Setup. The desktop will restart when recovery completes."
      );
      handoffError.isBootstrapFailure = true;
      handoffError.bootstrapHandedOff = true;
      bootstrapFailure = handoffError;
      throw handoffError;
    }
    try {
      broadcastBootstrapEvent({
        type: "manifest",
        stages: [],
        protocolVersion: null
      });
    } catch {
    }
    bootstrapAbortController = new AbortController();
    bootstrapRepairRequested = false;
    bootstrapRepairAttempt = 0;
    const bootstrapResult = await runBootstrap({
      installStamp: backend.installStamp,
      activeRoot: backend.activeRoot,
      sourceRepoRoot: SOURCE_REPO_ROOT,
      hermesHome: HERMES_HOME,
      logRoot: path22.join(HERMES_HOME, "logs"),
      abortSignal: bootstrapAbortController.signal,
      onEvent: (ev) => {
        try {
          rememberLog(`[bootstrap] ${JSON.stringify(ev)}`);
        } catch {
        }
        try {
          broadcastBootstrapEvent(ev);
        } catch {
        }
      },
      writeMarker: writeBootstrapMarker
    });
    bootstrapAbortController = null;
    if (bootstrapResult.cancelled) {
      const cancelledError = new Error("Hermes install was cancelled.");
      cancelledError.isBootstrapFailure = true;
      cancelledError.bootstrapCancelled = true;
      bootstrapFailure = cancelledError;
      throw cancelledError;
    }
    if (!bootstrapResult.ok) {
      const bootstrapError = new Error(
        `Hermes bootstrap failed${bootstrapResult.failedStage ? ` at stage '${bootstrapResult.failedStage}'` : ""}: ${bootstrapResult.error || "unknown error"}. Check ${path22.join(HERMES_HOME, "logs", "desktop.log")} for the full transcript.`
      );
      bootstrapError.isBootstrapFailure = true;
      bootstrapError.failedStage = bootstrapResult.failedStage || null;
      bootstrapFailure = bootstrapError;
      throw bootstrapError;
    }
    rememberLog("[bootstrap] bootstrap complete; marker written. Re-resolving backend.");
    return ensureRuntime(resolveHermesBackend(backend.args));
  }
  if (!isHermesSourceRoot(ACTIVE_HERMES_ROOT)) {
    throw new Error(
      `Hermes install at ${ACTIVE_HERMES_ROOT} is missing or incomplete. Reinstall via the desktop installer or scripts/install.ps1.`
    );
  }
  if (IS_WINDOWS3 && !findGitBash2()) {
    throw new Error(
      "Git for Windows is required for Hermes on Windows (provides Git Bash, which the agent's terminal tool uses). Install it from https://git-scm.com/download/win or run `winget install -e --id Git.Git`, then relaunch Hermes."
    );
  }
  const venvPython = getVenvPython(VENV_ROOT);
  if (!fileExists(venvPython)) {
    throw new Error(
      `Hermes venv missing at ${VENV_ROOT}. Re-run the desktop installer or \`scripts/install.ps1\` to rebuild it.`
    );
  }
  backend.command = getVenvPython(VENV_ROOT);
  backend.label = `Hermes at ${ACTIVE_HERMES_ROOT} (venv: ${VENV_ROOT})`;
  updateBootProgress({
    phase: "runtime.ready",
    message: "Hermes runtime is ready",
    progress: 82,
    running: true,
    error: null
  });
  return backend;
}
function multipartBody(upload) {
  const boundary = `----hermes-${crypto6.randomBytes(12).toString("hex")}`;
  const filename = String(upload.filename || "file").replace(/["\r\n]/g, "_");
  const body = Buffer.concat([
    Buffer.from(
      `--${boundary}\r
Content-Disposition: form-data; name="file"; filename="${filename}"\r
Content-Type: ${upload.contentType || "application/octet-stream"}\r
\r
`
    ),
    Buffer.from(upload.bytes),
    Buffer.from(`\r
--${boundary}--\r
`)
  ]);
  return { body, contentType: `multipart/form-data; boundary=${boundary}` };
}
function fetchJson(url, token, options = {}) {
  return new Promise((resolve, reject) => {
    const { body, contentType } = options.upload ? multipartBody(options.upload) : {
      body: options.body === void 0 ? void 0 : Buffer.from(JSON.stringify(options.body)),
      contentType: "application/json"
    };
    const parsed = new URL(url);
    const client = parsed.protocol === "https:" ? https3 : http2;
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`));
      return;
    }
    const req = client.request(
      parsed,
      {
        method: options.method || "GET",
        headers: {
          "Content-Type": contentType,
          "X-Hermes-Session-Token": token,
          // RFC 8252 native flow authenticates the gated gateway with a bearer
          // token instead of the loopback session-token header. When
          // ``options.bearer`` is set we send Authorization: Bearer <token>;
          // the gateway's OAuth gate verifies it via the provider stack with
          // no cookie involved.
          ...options.bearer ? { Authorization: `Bearer ${options.bearer}` } : {},
          ...body ? { "Content-Length": String(body.length) } : {}
        }
      },
      (res) => {
        const chunks = [];
        res.on("error", reject);
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          if ((res.statusCode || 500) >= 400) {
            reject(new Error(`${res.statusCode}: ${text || res.statusMessage}`));
            return;
          }
          if (!text) {
            resolve(null);
            return;
          }
          const looksHtml = /^\s*<(?:!doctype|html)/i.test(text);
          const contentType2 = String(res.headers["content-type"] || "");
          if (looksHtml || contentType2.includes("text/html")) {
            reject(
              new Error(
                `Expected JSON from ${url} but got HTML (status ${res.statusCode}). The endpoint is likely missing on the Hermes backend.`
              )
            );
            return;
          }
          try {
            resolve(JSON.parse(text));
          } catch {
            reject(new Error(`Invalid JSON from ${url} (status ${res.statusCode}): ${text.slice(0, 200)}`));
          }
        });
      }
    );
    req.on("error", reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`));
    });
    if (body) {
      req.write(body);
    }
    req.end();
  });
}
function fetchPublicJson(url, options = {}) {
  return new Promise((resolve, reject) => {
    const body = options.body === void 0 ? void 0 : Buffer.from(JSON.stringify(options.body));
    let parsed;
    try {
      parsed = new URL(url);
    } catch (error) {
      reject(new Error(`Invalid URL: ${error.message}`));
      return;
    }
    const client = parsed.protocol === "https:" ? https3 : http2;
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`));
      return;
    }
    const req = client.request(
      parsed,
      {
        method: options.method || "GET",
        headers: {
          "Content-Type": "application/json",
          ...body ? { "Content-Length": String(body.length) } : {}
        }
      },
      (res) => {
        const chunks = [];
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          if ((res.statusCode || 500) >= 400) {
            reject(new Error(`${res.statusCode}: ${text || res.statusMessage}`));
            return;
          }
          if (!text) {
            resolve(null);
            return;
          }
          const looksHtml = /^\s*<(?:!doctype|html)/i.test(text);
          const contentType = String(res.headers["content-type"] || "");
          if (looksHtml || contentType.includes("text/html")) {
            reject(
              new Error(
                `Expected JSON from ${url} but got HTML (status ${res.statusCode}). The endpoint is likely missing on the Hermes backend.`
              )
            );
            return;
          }
          try {
            resolve(JSON.parse(text));
          } catch {
            reject(new Error(`Invalid JSON from ${url} (status ${res.statusCode}): ${text.slice(0, 200)}`));
          }
        });
      }
    );
    req.on("error", reject);
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`));
    });
    if (body) {
      req.write(body);
    }
    req.end();
  });
}
function mimeTypeForPath(filePath) {
  const ext = path22.extname(filePath || "").toLowerCase();
  return MEDIA_MIME_TYPES[ext] || "application/octet-stream";
}
function extensionForMimeType(mimeType) {
  const type = String(mimeType || "").split(";")[0].trim().toLowerCase();
  if (type === "image/png") {
    return ".png";
  }
  if (type === "image/jpeg") {
    return ".jpg";
  }
  if (type === "image/gif") {
    return ".gif";
  }
  if (type === "image/webp") {
    return ".webp";
  }
  if (type === "image/bmp") {
    return ".bmp";
  }
  if (type === "image/svg+xml") {
    return ".svg";
  }
  return "";
}
function filenameFromUrl(rawUrl, fallback = "image") {
  try {
    const parsed = new URL(rawUrl);
    const base = path22.basename(decodeURIComponent(parsed.pathname || ""));
    return base && base.includes(".") ? base : fallback;
  } catch {
    return fallback;
  }
}
var titleCache = /* @__PURE__ */ new Map();
var titleInflight = /* @__PURE__ */ new Map();
var TITLE_CACHE_LIMIT = 500;
var TITLE_BYTE_BUDGET = 96 * 1024;
var TITLE_TIMEOUT_MS = 5e3;
var TITLE_MAX_REDIRECTS = 3;
var TITLE_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36";
var TITLE_ERROR_RE = /\b(access denied|attention required|captcha|error|forbidden|just a moment|request blocked|too many requests)\b/i;
var HTML_ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", "#39": "'" };
var RENDER_TITLE_MAX_CONCURRENT = 2;
var RENDER_TITLE_TIMEOUT_MS = 8e3;
var RENDER_TITLE_GRACE_MS = 700;
var RENDER_TITLE_BLOCKED_RESOURCES = /* @__PURE__ */ new Set([
  "cspReport",
  "font",
  "imageset",
  "media",
  "object",
  "ping",
  "stylesheet"
]);
var linkTitleSession = null;
var oauthSession = null;
var renderTitleInFlight = 0;
var renderTitleQueue = [];
function canonicalTitleCacheKey(rawUrl) {
  const value = String(rawUrl || "").trim();
  if (!value) {
    return "";
  }
  try {
    const url = new URL(value);
    const host = url.hostname.replace(/^www\./i, "").toLowerCase();
    const pathname = url.pathname === "/" ? "/" : url.pathname.replace(/\/+$/, "") || "/";
    return `${host}${pathname}${url.search || ""}`;
  } catch {
    return value;
  }
}
function cacheTitle(key, title) {
  if (titleCache.size >= TITLE_CACHE_LIMIT) {
    titleCache.delete(titleCache.keys().next().value);
  }
  titleCache.set(key, title);
}
function decodeHtmlEntities(value) {
  return value.replace(/&(amp|lt|gt|quot|apos|nbsp|#39);/gi, (_2, k2) => HTML_ENTITIES[k2.toLowerCase()] ?? "").replace(/&#x([0-9a-f]+);/gi, (_2, hex) => String.fromCodePoint(parseInt(hex, 16) || 32)).replace(/&#(\d+);/g, (_2, dec) => String.fromCodePoint(parseInt(dec, 10) || 32));
}
function parseHtmlTitle(html) {
  const raw = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1];
  return raw ? decodeHtmlEntities(raw).replace(/\s+/g, " ").trim() : "";
}
function fetchHtmlTitleWithCurl(rawUrl) {
  return new Promise((resolve) => {
    const url = String(rawUrl || "").trim();
    if (!url) {
      return resolve("");
    }
    const args = [
      "--silent",
      "--show-error",
      "--location",
      "--max-redirs",
      String(TITLE_MAX_REDIRECTS),
      "--max-time",
      String(Math.max(2, Math.ceil(TITLE_TIMEOUT_MS / 1e3))),
      "--connect-timeout",
      "4",
      "--user-agent",
      TITLE_USER_AGENT,
      "--header",
      "Accept: text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
      "--header",
      "Accept-Language: en-US,en;q=0.7",
      "--header",
      "Accept-Encoding: identity",
      "--raw",
      url
    ];
    const child = spawn5("curl", args, hiddenWindowsChildOptions({ stdio: ["ignore", "pipe", "ignore"] }));
    const chunks = [];
    let bytes = 0;
    child.stdout.on("data", (chunk) => {
      if (bytes >= TITLE_BYTE_BUDGET) {
        return;
      }
      const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
      const remaining = TITLE_BYTE_BUDGET - bytes;
      const next = buffer.length > remaining ? buffer.subarray(0, remaining) : buffer;
      chunks.push(next);
      bytes += next.length;
    });
    child.on("error", () => resolve(""));
    child.on("close", () => {
      if (!chunks.length) {
        return resolve("");
      }
      resolve(parseHtmlTitle(Buffer.concat(chunks).toString("utf8")));
    });
  });
}
function getLinkTitleSession() {
  if (linkTitleSession || !app2.isReady()) {
    return linkTitleSession;
  }
  linkTitleSession = session2.fromPartition("hermes:link-titles", { cache: false });
  linkTitleSession.webRequest.onBeforeRequest((details, callback) => {
    callback({ cancel: RENDER_TITLE_BLOCKED_RESOURCES.has(details.resourceType) });
  });
  guardLinkTitleSession(linkTitleSession);
  return linkTitleSession;
}
function dequeueRenderTitle() {
  while (renderTitleInFlight < RENDER_TITLE_MAX_CONCURRENT && renderTitleQueue.length) {
    const item = renderTitleQueue.shift();
    renderTitleInFlight += 1;
    runRenderTitleJob(item.url).then((title) => {
      renderTitleInFlight -= 1;
      item.resolve(title);
      dequeueRenderTitle();
    });
  }
}
function runRenderTitleJob(rawUrl) {
  return new Promise((resolve) => {
    if (!app2.isReady()) {
      return resolve("");
    }
    const partitionSession = getLinkTitleSession();
    if (!partitionSession) {
      return resolve("");
    }
    let settled = false;
    let window2 = null;
    let hardTimer = null;
    let graceTimer = null;
    const finish = (title) => {
      if (settled) {
        return;
      }
      settled = true;
      if (hardTimer) {
        clearTimeout(hardTimer);
      }
      if (graceTimer) {
        clearTimeout(graceTimer);
      }
      const value = (title || "").replace(/\s+/g, " ").trim();
      try {
        if (window2 && !window2.isDestroyed()) {
          window2.destroy();
        }
      } catch {
      }
      resolve(value);
    };
    try {
      window2 = createLinkTitleWindow(BrowserWindow2, partitionSession);
    } catch {
      return finish("");
    }
    const finishWithTitle = () => finish(readLinkTitleWindowTitle(window2));
    const scheduleGrace = () => {
      if (graceTimer) {
        clearTimeout(graceTimer);
      }
      graceTimer = setTimeout(finishWithTitle, RENDER_TITLE_GRACE_MS);
    };
    hardTimer = setTimeout(finishWithTitle, RENDER_TITLE_TIMEOUT_MS);
    window2.webContents.setUserAgent(TITLE_USER_AGENT);
    window2.webContents.on("page-title-updated", scheduleGrace);
    window2.webContents.on("did-finish-load", scheduleGrace);
    window2.webContents.on("did-fail-load", (_event, _code, _desc, _validatedURL, isMainFrame) => {
      if (isMainFrame) {
        finish("");
      }
    });
    window2.loadURL(rawUrl, {
      httpReferrer: "https://www.google.com/",
      userAgent: TITLE_USER_AGENT
    }).catch(() => finish(""));
  });
}
function fetchHtmlTitleWithRenderer(rawUrl) {
  return new Promise((resolve) => {
    renderTitleQueue.push({ resolve, url: rawUrl });
    dequeueRenderTitle();
  });
}
function usableTitle(value) {
  return value && !TITLE_ERROR_RE.test(value) ? value : "";
}
function fetchLinkTitle(rawUrl) {
  const url = String(rawUrl || "").trim();
  const key = canonicalTitleCacheKey(url);
  if (!key) {
    return Promise.resolve("");
  }
  if (titleCache.has(key)) {
    return Promise.resolve(titleCache.get(key));
  }
  if (titleInflight.has(key)) {
    return titleInflight.get(key);
  }
  const pending = fetchHtmlTitleWithCurl(url).catch(() => "").then((value) => usableTitle((value || "").slice(0, 240))).then(
    async (value) => value || usableTitle((await fetchHtmlTitleWithRenderer(url).catch(() => "") || "").slice(0, 240))
  ).then((clean) => {
    cacheTitle(key, clean);
    titleInflight.delete(key);
    return clean;
  });
  titleInflight.set(key, pending);
  return pending;
}
async function resourceBufferFromUrl(rawUrl) {
  if (!rawUrl) {
    throw new Error("Missing URL");
  }
  if (rawUrl.startsWith("data:")) {
    const match = rawUrl.match(/^data:([^;,]+)?(;base64)?,(.*)$/s);
    if (!match) {
      throw new Error("Invalid data URL");
    }
    const mimeType = match[1] || "application/octet-stream";
    const encoded = match[3] || "";
    const buffer = match[2] ? Buffer.from(encoded, "base64") : Buffer.from(decodeURIComponent(encoded), "utf8");
    return { buffer, mimeType };
  }
  if (/^file:/i.test(rawUrl)) {
    const { resolvedPath } = await resolveReadableFileForIpc(rawUrl, { purpose: "Image file" });
    const buffer = await fs19.promises.readFile(resolvedPath);
    return { buffer, mimeType: mimeTypeForPath(resolvedPath) };
  }
  const parsed = new URL(rawUrl);
  const client = parsed.protocol === "https:" ? https3 : http2;
  return new Promise((resolve, reject) => {
    const req = client.get(parsed, (res) => {
      if ((res.statusCode || 500) >= 400) {
        reject(new Error(`Failed to fetch ${rawUrl}: ${res.statusCode}`));
        res.resume();
        return;
      }
      const chunks = [];
      res.on("error", reject);
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => {
        resolve({
          buffer: Buffer.concat(chunks),
          mimeType: res.headers["content-type"] || "application/octet-stream"
        });
      });
    });
    req.on("error", reject);
  });
}
async function copyImageFromUrl(rawUrl) {
  const { buffer } = await resourceBufferFromUrl(rawUrl);
  const image = nativeImage2.createFromBuffer(buffer);
  if (image.isEmpty()) {
    throw new Error("Could not read image");
  }
  clipboard.writeImage(image);
}
async function saveImageFromUrl(rawUrl) {
  const { buffer, mimeType } = await resourceBufferFromUrl(rawUrl);
  const fallbackName = filenameFromUrl(rawUrl, `image${extensionForMimeType(mimeType) || ".png"}`);
  const result = await dialog.showSaveDialog(mainWindow, {
    title: "Save Image",
    defaultPath: fallbackName
  });
  if (result.canceled || !result.filePath) {
    return false;
  }
  await fs19.promises.writeFile(result.filePath, buffer);
  return true;
}
async function writeComposerImage(buffer, ext = ".png") {
  const rawExt = String(ext || ".png").trim().toLowerCase();
  const normalizedExt = rawExt.startsWith(".") ? rawExt : `.${rawExt}`;
  const safeExt = /^\.[a-z0-9]{1,5}$/.test(normalizedExt) ? normalizedExt : ".png";
  const dir = path22.join(app2.getPath("userData"), "composer-images");
  await fs19.promises.mkdir(dir, { recursive: true });
  const stamp = (/* @__PURE__ */ new Date()).toISOString().replace(/[:.]/g, "-").replace("T", "_").replace("Z", "");
  const random = crypto6.randomBytes(3).toString("hex");
  const filePath = path22.join(dir, `composer_${stamp}_${random}${safeExt}`);
  await fs19.promises.writeFile(filePath, buffer);
  return filePath;
}
function previewLabelForUrl(url) {
  return `${url.host}${url.pathname === "/" ? "" : url.pathname}`;
}
function expandUserPath(filePath) {
  const value = String(filePath || "").trim();
  if (value === "~") {
    return app2.getPath("home");
  }
  if (value.startsWith(`~${path22.sep}`) || value.startsWith("~/")) {
    return path22.join(app2.getPath("home"), value.slice(2));
  }
  return value;
}
async function previewFileTarget(rawTarget, baseDir) {
  const raw = String(rawTarget || "").trim();
  const base = baseDir ? path22.resolve(expandUserPath(baseDir)) : resolveHermesCwd();
  let resolved = resolveRequestedPathForIpc(/^file:/i.test(raw) ? raw : expandUserPath(raw), {
    baseDir: base,
    purpose: "Preview target"
  });
  if (directoryExists(resolved)) {
    resolved = path22.join(resolved, "index.html");
  }
  const ext = path22.extname(resolved).toLowerCase();
  if (!fileExists(resolved)) {
    return null;
  }
  ;
  ({ resolvedPath: resolved } = await resolveReadableFileForIpc(resolved, { purpose: "Preview target" }));
  const mimeType = mimeTypeForPath(resolved);
  const metadata = previewFileMetadata(resolved, mimeType);
  const isHtml = PREVIEW_HTML_EXTENSIONS.has(ext);
  const isImage = mimeType.startsWith("image/");
  const previewKind = isHtml ? "html" : isImage ? "image" : metadata.binary ? "binary" : "text";
  return {
    binary: metadata.binary,
    byteSize: metadata.byteSize,
    kind: "file",
    large: metadata.large,
    label: path22.basename(resolved),
    language: PREVIEW_LANGUAGE_BY_EXT[ext] || "text",
    mimeType,
    path: resolved,
    previewKind,
    source: raw,
    url: pathToFileURL3(resolved).toString()
  };
}
function previewUrlTarget(rawTarget) {
  const raw = String(rawTarget || "").trim();
  const url = new URL(raw);
  if (!["http:", "https:"].includes(url.protocol)) {
    return null;
  }
  if (!LOCAL_PREVIEW_HOSTS.has(url.hostname.toLowerCase())) {
    return null;
  }
  if (url.hostname === "0.0.0.0") {
    url.hostname = "127.0.0.1";
  }
  return {
    kind: "url",
    label: previewLabelForUrl(url),
    source: raw,
    url: url.toString()
  };
}
async function normalizePreviewTarget(rawTarget, baseDir) {
  const raw = String(rawTarget || "").trim();
  if (!raw) {
    return null;
  }
  try {
    if (/^https?:\/\//i.test(raw)) {
      return previewUrlTarget(raw);
    }
    return await previewFileTarget(raw, baseDir);
  } catch {
    return null;
  }
}
async function filePathFromPreviewUrl(rawUrl) {
  const { resolvedPath } = await resolveReadableFileForIpc(String(rawUrl || ""), { purpose: "Preview file" });
  return resolvedPath;
}
function sendPreviewFileChanged(payload) {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  webContents.send("hermes:preview-file-changed", payload);
}
async function watchPreviewFile(rawUrl) {
  const filePath = await filePathFromPreviewUrl(rawUrl);
  const watchDir = path22.dirname(filePath);
  const targetName = path22.basename(filePath);
  const id = crypto6.randomBytes(12).toString("base64url");
  let timer = null;
  const watcher = fs19.watch(watchDir, (_eventType, filename) => {
    const changedName = filename ? path22.basename(String(filename)) : "";
    if (changedName && changedName !== targetName) {
      return;
    }
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = null;
      if (!fileExists(filePath)) {
        return;
      }
      sendPreviewFileChanged({ id, path: filePath, url: pathToFileURL3(filePath).toString() });
    }, PREVIEW_WATCH_DEBOUNCE_MS);
  });
  previewWatchers.set(id, {
    close: () => {
      if (timer) {
        clearTimeout(timer);
      }
      watcher.close();
    }
  });
  return { id, path: filePath };
}
function stopPreviewFileWatch(id) {
  const watcher = previewWatchers.get(id);
  if (!watcher) {
    return false;
  }
  watcher.close();
  previewWatchers.delete(id);
  return true;
}
function closePreviewWatchers() {
  for (const id of previewWatchers.keys()) {
    stopPreviewFileWatch(id);
  }
}
function watchDirectory(rawDir) {
  const watchDir = path22.resolve(String(rawDir || ""));
  if (!fs19.existsSync(watchDir) || !fs19.statSync(watchDir).isDirectory()) {
    throw new Error(`Not a directory: ${watchDir}`);
  }
  const id = crypto6.randomBytes(12).toString("base64url");
  let timer = null;
  const watcher = fs19.watch(watchDir, () => {
    if (timer) {
      clearTimeout(timer);
    }
    timer = setTimeout(() => {
      timer = null;
      sendPreviewFileChanged({ id, path: watchDir, url: pathToFileURL3(watchDir).toString() });
    }, PREVIEW_WATCH_DEBOUNCE_MS);
  });
  previewWatchers.set(id, {
    close: () => {
      if (timer) {
        clearTimeout(timer);
      }
      watcher.close();
    }
  });
  return { id, path: watchDir };
}
var gatewayAuthProvidersCache = /* @__PURE__ */ new Map();
async function gatewayAuthProviders(baseUrl) {
  const cached = gatewayAuthProvidersCache.get(baseUrl);
  if (cached) {
    return cached;
  }
  let providers = [];
  try {
    const body = await fetchPublicJson(`${baseUrl}/api/auth/providers`, { timeoutMs: 8e3 });
    if (Array.isArray(body?.providers)) {
      providers = body.providers.filter((p2) => p2 && typeof p2 === "object").map((p2) => ({ name: String(p2.name || ""), supportsPassword: Boolean(p2.supports_password) })).filter((p2) => p2.name);
    }
  } catch {
  }
  gatewayAuthProvidersCache.set(baseUrl, providers);
  return providers;
}
async function buildReadinessHealthProbe(baseUrl, authMode, token) {
  const nativeAt = authMode === "oauth" ? await ensureNativeAccessToken(baseUrl).catch(() => null) : null;
  const probeAuth = resolveReadinessProbeAuth(authMode, nativeAt, token);
  if (probeAuth.kind === "bearer") {
    return {
      // fetchJson takes the bearer via `options.bearer` — a raw `headers`
      // option is ignored, so passing one here would silently probe
      // uncredentialed and reintroduce the 401 loop.
      probeHealth: (url, options = {}) => fetchJson(url, null, { ...options, bearer: probeAuth.token }),
      probeIsCredentialed: true
    };
  }
  if (probeAuth.kind === "cookie") {
    return {
      probeHealth: (url, options = {}) => fetchJsonViaOauthSession(url, options),
      probeIsCredentialed: true
    };
  }
  if (probeAuth.kind === "token" && probeAuth.token) {
    return {
      probeHealth: (url, options = {}) => fetchJson(url, probeAuth.token, options),
      probeIsCredentialed: true
    };
  }
  return { probeHealth: fetchPublicJson, probeIsCredentialed: false };
}
async function waitForHermes(baseUrl, token, signal, authMode) {
  const { probeHealth, probeIsCredentialed } = await buildReadinessHealthProbe(baseUrl, authMode, token);
  return waitForHermesReady(baseUrl, {
    token,
    signal,
    fetchPublicJson,
    fetchJson: probeIsCredentialed ? (url, _token, options) => probeHealth(url, options) : fetchJson,
    probeHealth,
    probeIsCredentialed
  });
}
function getWindowButtonPosition() {
  if (!IS_MAC) {
    return null;
  }
  return mainWindow?.getWindowButtonPosition?.() || WINDOW_BUTTON_POSITION;
}
function getNativeOverlayWidth() {
  return nativeOverlayWidth({ isWindows: IS_WINDOWS3, isWsl: IS_WSL, isMac: IS_MAC });
}
function getWindowState(win = mainWindow) {
  return {
    isFullscreen: Boolean(win?.isFullScreen?.()),
    isMinimized: Boolean(win?.isMinimized?.()),
    isVisible: Boolean(win?.isVisible?.()),
    nativeOverlayWidth: getNativeOverlayWidth(),
    windowButtonPosition: getWindowButtonPosition()
  };
}
function sendBackendExit(payload) {
  if (softRehomeInProgress) {
    return;
  }
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  webContents.send("hermes:backend-exit", payload);
}
function sendClosePreviewRequested() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  webContents.send("hermes:close-preview-requested");
}
function sendOpenFolderRequested() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  const webContents = mainWindow.webContents;
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  webContents.send("hermes:open-folder-requested");
}
function sendPowerResume() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  webContents.send("hermes:power-resume");
}
var powerResumeRegistered = false;
var onBatteryPower = null;
ipcMain.handle("hermes:power-battery:get", () => onBatteryPower === true);
function broadcastBatteryState(next) {
  if (onBatteryPower === next) {
    return;
  }
  onBatteryPower = next;
  for (const win of BrowserWindow2.getAllWindows()) {
    const { webContents } = win;
    if (webContents && !webContents.isDestroyed()) {
      webContents.send("hermes:power-battery", next);
    }
  }
}
function registerPowerResumeListeners() {
  if (powerResumeRegistered) {
    return;
  }
  powerResumeRegistered = true;
  try {
    powerMonitor.on("resume", sendPowerResume);
    powerMonitor.on("unlock-screen", sendPowerResume);
    powerMonitor.on("on-battery", () => broadcastBatteryState(true));
    powerMonitor.on("on-ac", () => broadcastBatteryState(false));
    onBatteryPower = powerMonitor.isOnBatteryPower();
  } catch {
  }
}
function getAppIconPath() {
  return APP_ICON_PATHS.find(fileExists);
}
function sendOpenUpdatesRequested() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  webContents.send("hermes:open-updates");
  if (!mainWindow.isVisible()) {
    mainWindow.show();
  }
  mainWindow.focus();
}
function sendWindowStateChanged(nextIsFullscreen, target2 = mainWindow) {
  if (!target2 || target2.isDestroyed()) {
    return;
  }
  const { webContents } = target2;
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  const state = getWindowState(target2);
  if (typeof nextIsFullscreen === "boolean") {
    state.isFullscreen = nextIsFullscreen;
  }
  webContents.send("hermes:window-state-changed", state);
}
function buildApplicationMenu() {
  const template = [];
  const checkForUpdatesItem = {
    label: "Check for Updates\u2026",
    click: () => sendOpenUpdatesRequested()
  };
  if (IS_MAC) {
    template.push({
      label: APP_NAME,
      submenu: [
        { label: `About ${APP_NAME}`, click: () => showAboutPanelFresh() },
        checkForUpdatesItem,
        { type: "separator" },
        { role: "services" },
        { type: "separator" },
        { role: "hide" },
        { role: "hideOthers" },
        { role: "unhide" },
        { type: "separator" },
        { role: "quit" }
      ]
    });
  }
  template.push({
    label: "File",
    submenu: [
      // No accelerator: ⌘⇧N is a rebindable renderer keybind (session.newWindow);
      // a menu accelerator would fight the rebind panel and (on macOS) be
      // swallowed before the renderer sees it. Here purely for discoverability.
      { click: () => createInstanceWindow(), label: "New Window" },
      // Same no-accelerator rationale: ⌘O is the rebindable renderer keybind
      // (workspace.openFolder). Clicking runs the same open-folder-as-project
      // flow through the renderer.
      { click: () => sendOpenFolderRequested(), label: "Open Folder\u2026" },
      { type: "separator" },
      IS_MAC ? {
        // NO accelerator: on macOS a registered ⌘W is consumed by the OS
        // menu before the web contents ever sees it (and registerAccelerator
        // false is a no-op on mac — electron#18295). Leaving it off lets the
        // `before-input-event` handler below intercept ⌘W and route it to the
        // renderer's close-active-tab. Clicking the item still closes the tab
        // (or window) via the same request.
        click: () => sendClosePreviewRequested(),
        label: "Close"
      } : { role: "quit" }
    ]
  });
  template.push({
    label: "Edit",
    submenu: [
      { role: "undo" },
      { role: "redo" },
      { type: "separator" },
      { role: "cut" },
      { role: "copy" },
      { role: "paste" },
      // ⌘⇧V is only wired up by this item existing: an accelerator with no menu
      // entry is never translated into an editor command, so the chord was a
      // no-op in every input in the app. The composer inserts plain text on
      // every paste anyway, so this is the same result as ⌘V there — it's the
      // terminal, preview, and other editable surfaces that need the strip.
      { role: "pasteAndMatchStyle" },
      { role: "delete" },
      { role: "selectAll" }
    ]
  });
  template.push({
    label: "View",
    submenu: [
      { role: "reload" },
      { role: "forceReload" },
      { role: "toggleDevTools" },
      { type: "separator" },
      {
        label: "Actual Size",
        accelerator: "CommandOrControl+0",
        click: () => {
          setAndPersistZoomLevel(mainWindow, DEFAULT_ZOOM_LEVEL);
        }
      },
      {
        label: "Zoom In",
        accelerator: "CommandOrControl+Plus",
        click: () => {
          if (mainWindow && !mainWindow.isDestroyed()) {
            setAndPersistZoomLevel(mainWindow, mainWindow.webContents.getZoomLevel() + ZOOM_STEP);
          }
        }
      },
      {
        label: "Zoom Out",
        accelerator: "CommandOrControl+-",
        click: () => {
          if (mainWindow && !mainWindow.isDestroyed()) {
            setAndPersistZoomLevel(mainWindow, mainWindow.webContents.getZoomLevel() - ZOOM_STEP);
          }
        }
      },
      { type: "separator" },
      { role: "togglefullscreen" }
    ]
  });
  template.push({
    label: "Window",
    submenu: IS_MAC ? [{ role: "minimize" }, { role: "zoom" }, { role: "front" }] : [{ role: "minimize" }, { role: "close" }]
  });
  template.push({
    label: "Help",
    role: "help",
    submenu: [checkForUpdatesItem]
  });
  return Menu2.buildFromTemplate(template);
}
function toggleDevTools(window2) {
  const { webContents } = window2;
  if (webContents.isDevToolsOpened()) {
    webContents.closeDevTools();
  } else {
    webContents.openDevTools({ mode: "detach" });
  }
}
function installDevToolsShortcut(window2) {
  window2.webContents.on("before-input-event", (event, input) => {
    const key = input.key.toLowerCase();
    const isInspectShortcut = input.key === "F12" || IS_MAC && input.meta && input.alt && key === "i" || !IS_MAC && input.control && input.shift && key === "i";
    if (!isInspectShortcut) {
      return;
    }
    event.preventDefault();
    toggleDevTools(window2);
  });
}
function installPreviewShortcut(window2) {
  window2.webContents.on("before-input-event", (event, input) => {
    const key = String(input.key || "").toLowerCase();
    const isCloseTabShortcut = key === "w" && (IS_MAC ? input.meta : input.control) && !input.alt && !input.shift;
    if (!isCloseTabShortcut) {
      return;
    }
    event.preventDefault();
    sendClosePreviewRequested();
  });
}
function setAndPersistZoomLevel(window2, zoomLevel) {
  if (!window2 || window2.isDestroyed()) {
    return;
  }
  const next = applyZoomLevel(window2.webContents, zoomLevel);
  writeZoomState(next);
  window2.webContents.executeJavaScript(
    `try { localStorage.setItem(${JSON.stringify(ZOOM_STORAGE_KEY)}, ${JSON.stringify(String(next))}) } catch {
      void 0
    }`
  ).catch((error) => rememberLog(`[zoom] persist failed: ${error?.message || error}`));
}
function restorePersistedZoomLevel(window2) {
  if (!window2 || window2.isDestroyed()) {
    return;
  }
  const saved = readZoomState();
  if (saved != null) {
    applyZoomLevel(window2.webContents, saved);
    return;
  }
  applyZoomLevel(window2.webContents, DEFAULT_ZOOM_LEVEL);
  window2.webContents.executeJavaScript(
    `(() => { try { return localStorage.getItem(${JSON.stringify(ZOOM_STORAGE_KEY)}) } catch { return null } })()`
  ).then((stored) => {
    if (!window2 || window2.isDestroyed()) {
      return;
    }
    const level = stored == null ? DEFAULT_ZOOM_LEVEL : Number(stored);
    const applied = applyZoomLevel(window2.webContents, level);
    writeZoomState(applied);
  }).catch((error) => rememberLog(`[zoom] restore failed: ${error?.message || error}`));
}
function installZoomShortcuts(window2) {
  window2.webContents.on("before-input-event", (event, input) => {
    const mod = IS_MAC ? input.meta : input.control;
    if (!mod || input.alt) {
      return;
    }
    const key = input.key;
    if (key === "0") {
      if (input.shift) {
        return;
      }
      event.preventDefault();
      setAndPersistZoomLevel(window2, DEFAULT_ZOOM_LEVEL);
    } else if (key === "=" || key === "+") {
      event.preventDefault();
      setAndPersistZoomLevel(window2, window2.webContents.getZoomLevel() + ZOOM_STEP);
    } else if (key === "-") {
      if (input.shift) {
        return;
      }
      event.preventDefault();
      setAndPersistZoomLevel(window2, window2.webContents.getZoomLevel() - ZOOM_STEP);
    }
  });
  window2.webContents.on("zoom-changed", (event, zoomDirection) => {
    event.preventDefault();
    const delta = zoomDirection === "in" ? ZOOM_STEP : -ZOOM_STEP;
    setAndPersistZoomLevel(window2, window2.webContents.getZoomLevel() + delta);
  });
}
function installContextMenu(window2) {
  window2.webContents.on("context-menu", (_event, params) => {
    const template = [];
    const hasSelection = Boolean(params.selectionText?.trim());
    const hasImage = params.mediaType === "image" && Boolean(params.srcURL);
    const hasLink = Boolean(params.linkURL);
    const isEditable = Boolean(params.isEditable);
    if (hasImage) {
      template.push(
        {
          label: "Open Image",
          click: () => {
            if (params.srcURL && !params.srcURL.startsWith("data:")) {
              openExternalUrl(params.srcURL);
            }
          },
          enabled: !params.srcURL.startsWith("data:")
        },
        {
          label: "Copy Image",
          click: () => {
            void copyImageFromUrl(params.srcURL).catch((error) => rememberLog(`Copy image failed: ${error.message}`));
          }
        },
        {
          label: "Copy Image Address",
          click: () => clipboard.writeText(params.srcURL)
        },
        {
          label: "Save Image As...",
          click: () => {
            void saveImageFromUrl(params.srcURL).catch((error) => rememberLog(`Save image failed: ${error.message}`));
          }
        }
      );
    }
    if (hasLink) {
      if (template.length) {
        template.push({ type: "separator" });
      }
      template.push(
        {
          label: "Open Link",
          click: () => openExternalUrl(params.linkURL)
        },
        {
          label: "Copy Link",
          click: () => clipboard.writeText(params.linkURL)
        }
      );
    }
    const suggestions = Array.isArray(params.dictionarySuggestions) ? params.dictionarySuggestions : [];
    if (isEditable && params.misspelledWord && suggestions.length > 0) {
      if (template.length) {
        template.push({ type: "separator" });
      }
      for (const suggestion of suggestions.slice(0, 5)) {
        template.push({
          label: suggestion,
          click: () => window2.webContents.replaceMisspelling(suggestion)
        });
      }
      template.push({ type: "separator" });
      template.push({
        label: "Add to dictionary",
        click: () => window2.webContents.session.addWordToSpellCheckerDictionary(params.misspelledWord)
      });
    }
    if (hasSelection || isEditable) {
      if (template.length) {
        template.push({ type: "separator" });
      }
      if (isEditable) {
        template.push(
          { role: "cut", enabled: params.editFlags.canCut },
          { role: "copy", enabled: params.editFlags.canCopy },
          { role: "paste", enabled: params.editFlags.canPaste },
          { type: "separator" },
          { role: "selectAll", enabled: params.editFlags.canSelectAll }
        );
      } else {
        template.push({ role: "copy", enabled: params.editFlags.canCopy });
      }
    }
    if (!template.length) {
      return;
    }
    Menu2.buildFromTemplate(template).popup({ window: window2 });
  });
}
function isMediaCapturePermission(permission, details) {
  if (permission === "audioCapture" || permission === "videoCapture") {
    return true;
  }
  if (permission !== "media") {
    return false;
  }
  const mediaTypes = details?.mediaTypes;
  if (!Array.isArray(mediaTypes) || mediaTypes.length === 0) {
    return true;
  }
  return mediaTypes.includes("audio") || mediaTypes.includes("video");
}
function installMediaPermissions() {
  session2.defaultSession.setPermissionRequestHandler((_webContents, permission, callback, details) => {
    callback(isMediaCapturePermission(permission, details));
  });
  session2.defaultSession.setPermissionCheckHandler((_webContents, permission) => {
    return permission === "media" || permission === "audioCapture" || permission === "videoCapture";
  });
}
var OAUTH_SESSION_PARTITION = "persist:hermes-remote-oauth";
function getOauthSession() {
  if (oauthSession || !app2.isReady()) {
    return oauthSession;
  }
  oauthSession = session2.fromPartition(OAUTH_SESSION_PARTITION);
  return oauthSession;
}
var oauthCookieWarmup = null;
function warmOauthCookieStore() {
  if (oauthCookieWarmup) {
    return oauthCookieWarmup;
  }
  oauthCookieWarmup = (async () => {
    const sess = getOauthSession();
    if (!sess) {
      oauthCookieWarmup = null;
      return;
    }
    try {
      sess.flushStorageData?.();
      await sess.cookies.get({});
    } catch {
    }
  })();
  return oauthCookieWarmup;
}
async function hasOauthSessionCookie(baseUrl) {
  const sess = getOauthSession();
  if (!sess) {
    return false;
  }
  const parsed = new URL(baseUrl);
  try {
    const cookies = await sess.cookies.get({ url: baseUrl });
    return cookiesHaveSession(cookies);
  } catch {
    try {
      const cookies = await sess.cookies.get({ domain: parsed.hostname });
      return cookiesHaveSession(cookies);
    } catch {
      return false;
    }
  }
}
async function hasLiveOauthSession(baseUrl) {
  const sess = getOauthSession();
  if (!sess) {
    return false;
  }
  const parsed = new URL(baseUrl);
  const readLive = async () => {
    try {
      const cookies = await sess.cookies.get({ url: baseUrl });
      return cookiesHaveLiveSession(cookies);
    } catch {
      try {
        const cookies = await sess.cookies.get({ domain: parsed.hostname });
        return cookiesHaveLiveSession(cookies);
      } catch {
        return false;
      }
    }
  };
  if (await readLive()) {
    return true;
  }
  await warmOauthCookieStore();
  for (const delayMs of [30, 60, 90]) {
    if (await readLive()) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, delayMs));
  }
  return readLive();
}
async function clearOauthSession(baseUrl) {
  const sess = getOauthSession();
  if (!sess) {
    return;
  }
  try {
    const cookies = await sess.cookies.get(baseUrl ? { url: baseUrl } : {});
    await Promise.all(
      cookies.map((c3) => {
        const scheme = c3.secure ? "https" : "http";
        const cookieUrl = `${scheme}://${c3.domain.replace(/^\./, "")}${c3.path || "/"}`;
        return sess.cookies.remove(cookieUrl, c3.name).catch(() => void 0);
      })
    );
  } catch {
  }
}
function openOauthLoginWindow(baseUrl, { silent = false } = {}) {
  return new Promise((resolve, reject) => {
    if (!app2.isReady()) {
      reject(new Error("Desktop is not ready to start an OAuth login."));
      return;
    }
    const sess = getOauthSession();
    if (!sess) {
      reject(new Error("OAuth session partition is unavailable."));
      return;
    }
    let settled = false;
    let win = null;
    let pollTimer = null;
    let revealTimer = null;
    const finish = (err) => {
      if (settled) {
        return;
      }
      settled = true;
      if (pollTimer) {
        clearInterval(pollTimer);
      }
      if (revealTimer) {
        clearTimeout(revealTimer);
      }
      try {
        if (win && !win.isDestroyed()) {
          win.destroy();
        }
      } catch {
      }
      if (err) {
        reject(err);
      } else {
        resolve({ baseUrl, ok: true });
      }
    };
    const checkCookie = async () => {
      if (settled) {
        return;
      }
      if (await hasOauthSessionCookie(baseUrl)) {
        finish(null);
      }
    };
    try {
      win = new BrowserWindow2({
        width: 520,
        height: 720,
        title: silent ? "Connecting to Hermes Cloud agent\u2026" : "Sign in to Hermes gateway",
        autoHideMenuBar: true,
        // Silent cascade: start HIDDEN. The auto-SSO 302 chain completes in
        // well under a second, so the window normally never needs to show. We
        // only reveal it as a fallback if the cascade DOESN'T complete quickly
        // (e.g. the portal session lapsed and the gate fell through to the
        // interactive chooser) — see the reveal timer below.
        show: !silent,
        webPreferences: {
          contextIsolation: true,
          nodeIntegration: false,
          sandbox: true,
          session: sess,
          webSecurity: true
        }
      });
    } catch (error) {
      finish(error instanceof Error ? error : new Error(String(error)));
      return;
    }
    win.webContents.on("did-navigate", () => void checkCookie());
    win.webContents.on("did-redirect-navigation", () => void checkCookie());
    win.webContents.on("did-frame-navigate", () => void checkCookie());
    pollTimer = setInterval(() => void checkCookie(), 750);
    if (silent && win) {
      revealTimer = setTimeout(() => {
        try {
          if (!settled && win && !win.isDestroyed() && !win.isVisible()) {
            win.show();
          }
        } catch {
        }
      }, 2500);
    }
    win.on("closed", () => {
      if (!settled) {
        finish(new Error("Login window closed before authentication completed."));
      }
    });
    const normalizedBase = normalizeRemoteBaseUrl(baseUrl);
    const loginUrl = silent ? `${normalizedBase}/` : `${normalizedBase}/login`;
    win.loadURL(loginUrl).catch((error) => {
      finish(error instanceof Error ? error : new Error(String(error)));
    });
  });
}
function fetchJsonViaOauthSession(url, options = {}) {
  return new Promise((resolve, reject) => {
    const sess = getOauthSession();
    if (!sess) {
      reject(new Error("OAuth session partition is unavailable."));
      return;
    }
    let parsed;
    try {
      parsed = new URL(url);
    } catch (error) {
      reject(new Error(`Invalid URL: ${error.message}`));
      return;
    }
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      reject(new Error(`Unsupported Hermes backend URL protocol: ${parsed.protocol}`));
      return;
    }
    const body = serializeJsonBody(options.body);
    const timeoutMs = resolveTimeoutMs(options.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS);
    const request2 = electronNet.request({
      method: options.method || "GET",
      url,
      session: sess,
      useSessionCookies: true,
      redirect: "follow"
    });
    setJsonRequestHeaders(request2);
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      try {
        request2.abort();
      } catch {
      }
      reject(new Error(`Timed out connecting to Hermes backend after ${timeoutMs}ms`));
    }, timeoutMs);
    request2.on("response", (res) => {
      const chunks = [];
      res.on("data", (chunk) => chunks.push(Buffer.from(chunk)));
      res.on("end", () => {
        if (timedOut) {
          return;
        }
        clearTimeout(timer);
        const text = Buffer.concat(chunks).toString("utf8");
        const statusCode = res.statusCode || 500;
        if (statusCode >= 400) {
          const err = new Error(`${statusCode}: ${text || ""}`);
          err.statusCode = statusCode;
          reject(err);
          return;
        }
        if (!text) {
          resolve(null);
          return;
        }
        const looksHtml = /^\s*<(?:!doctype|html)/i.test(text);
        const contentType = String(res.headers["content-type"] || res.headers["Content-Type"] || "");
        if (looksHtml || contentType.includes("text/html")) {
          reject(new Error(`Expected JSON from ${url} but got HTML (status ${statusCode}).`));
          return;
        }
        try {
          resolve(JSON.parse(text));
        } catch {
          reject(new Error(`Invalid JSON from ${url} (status ${statusCode}): ${text.slice(0, 200)}`));
        }
      });
    });
    request2.on("error", (error) => {
      if (timedOut) {
        return;
      }
      clearTimeout(timer);
      reject(error);
    });
    if (body) {
      request2.write(body);
    }
    request2.end();
  });
}
var _nativeTokens = /* @__PURE__ */ new Map();
function _nativeTokenStorePath() {
  return path22.join(app2.getPath("userData"), "native-oauth-tokens.json");
}
function _nativeTokenStoreIo() {
  return {
    encrypt: encryptDesktopSecret2,
    decrypt: decryptDesktopSecret,
    readStoreText: () => fs19.readFileSync(_nativeTokenStorePath(), "utf8"),
    writeStoreText: (text) => {
      fs19.mkdirSync(path22.dirname(_nativeTokenStorePath()), { recursive: true });
      fs19.writeFileSync(_nativeTokenStorePath(), text, { mode: 384 });
    },
    rememberLog
  };
}
function _persistNativeTokens(baseUrl, tokens) {
  persistNativeTokenSet(baseUrl, tokens, _nativeTokenStoreIo());
}
function _loadNativeTokens(baseUrl) {
  const cached = _nativeTokens.get(baseUrl);
  if (cached) {
    return cached;
  }
  const tokens = loadNativeTokenSet(baseUrl, _nativeTokenStoreIo());
  if (tokens) {
    _nativeTokens.set(baseUrl, tokens);
  }
  return tokens;
}
function _storeNativeTokens(baseUrl, tokens) {
  _nativeTokens.set(baseUrl, tokens);
  _persistNativeTokens(baseUrl, tokens);
}
function _clearNativeTokens(baseUrl) {
  _nativeTokens.delete(baseUrl);
  _persistNativeTokens(baseUrl, null);
}
function hasNativeSession(baseUrl) {
  return _loadNativeTokens(baseUrl) !== null;
}
function postJsonNoAuth(url, body, opts = {}) {
  return fetchJson(url, null, { method: "POST", body: resolveJsonBody(body), ...opts });
}
async function ensureNativeAccessToken(baseUrl) {
  const tokens = _loadNativeTokens(baseUrl);
  if (!tokens) {
    return null;
  }
  if (!tokenNeedsRefresh(tokens, Math.floor(Date.now() / 1e3))) {
    return tokens.accessToken;
  }
  if (!tokens.refreshToken) {
    _clearNativeTokens(baseUrl);
    return null;
  }
  try {
    const body = await postJsonNoAuth(
      nativeRefreshUrl(baseUrl),
      { refresh_token: tokens.refreshToken, provider: tokens.provider },
      { timeoutMs: 1e4 }
    );
    const rotated = parseTokenResponse(body);
    _storeNativeTokens(baseUrl, rotated);
    return rotated.accessToken;
  } catch (error) {
    if (error && error.statusCode === 401) {
      _clearNativeTokens(baseUrl);
      return null;
    }
    throw error;
  }
}
async function mintGatewayWsTicket(baseUrl) {
  const nativeAt = await ensureNativeAccessToken(baseUrl).catch(() => null);
  if (nativeAt) {
    const body2 = await fetchJson(`${baseUrl}/api/auth/ws-ticket`, null, {
      method: "POST",
      timeoutMs: 8e3,
      bearer: nativeAt
    });
    const ticket2 = body2?.ticket;
    if (!ticket2 || typeof ticket2 !== "string") {
      throw new Error("Gateway did not return a WS ticket.");
    }
    return ticket2;
  }
  const body = await fetchJsonViaOauthSession(`${baseUrl}/api/auth/ws-ticket`, {
    method: "POST",
    timeoutMs: 8e3
  });
  const ticket = body?.ticket;
  if (!ticket || typeof ticket !== "string") {
    throw new Error("Gateway did not return a WS ticket.");
  }
  return ticket;
}
async function freshGatewayWsUrl(profile) {
  const connection = await ensureBackend(profile);
  if (connection.authMode === "oauth") {
    const ticket = await mintGatewayWsTicket(connection.baseUrl);
    return buildGatewayWsUrlWithTicket(connection.baseUrl, ticket);
  }
  return connection.wsUrl;
}
var DEFAULT_NOUS_PORTAL_URL = "https://portal.nousresearch.com";
function resolvePortalBaseUrl() {
  const raw = process.env.HERMES_PORTAL_BASE_URL || process.env.NOUS_PORTAL_BASE_URL || DEFAULT_NOUS_PORTAL_URL;
  return String(raw).trim().replace(/\/+$/, "");
}
async function hasLivePortalSession() {
  const sess = getOauthSession();
  if (!sess) {
    return false;
  }
  const portalBaseUrl = resolvePortalBaseUrl();
  const parsed = new URL(portalBaseUrl);
  try {
    const cookies = await sess.cookies.get({ url: portalBaseUrl });
    return cookiesHavePrivySession(cookies);
  } catch {
    try {
      const cookies = await sess.cookies.get({ domain: parsed.hostname });
      return cookiesHavePrivySession(cookies);
    } catch {
      return false;
    }
  }
}
function openPortalLoginWindow() {
  const portalBaseUrl = resolvePortalBaseUrl();
  return new Promise((resolve, reject) => {
    if (!app2.isReady()) {
      reject(new Error("Desktop is not ready to start a Hermes Cloud sign-in."));
      return;
    }
    const sess = getOauthSession();
    if (!sess) {
      reject(new Error("OAuth session partition is unavailable."));
      return;
    }
    let settled = false;
    let win = null;
    let pollTimer = null;
    const finish = (err) => {
      if (settled) {
        return;
      }
      settled = true;
      if (pollTimer) {
        clearInterval(pollTimer);
      }
      try {
        if (win && !win.isDestroyed()) {
          win.destroy();
        }
      } catch {
      }
      if (err) {
        reject(err);
      } else {
        resolve({ portalBaseUrl, ok: true });
      }
    };
    const checkCookie = async () => {
      if (settled) {
        return;
      }
      if (await hasLivePortalSession()) {
        finish(null);
      }
    };
    try {
      win = new BrowserWindow2({
        width: 520,
        height: 720,
        title: "Sign in to Hermes Cloud",
        autoHideMenuBar: true,
        webPreferences: {
          contextIsolation: true,
          nodeIntegration: false,
          sandbox: true,
          session: sess,
          webSecurity: true
        }
      });
    } catch (error) {
      finish(error instanceof Error ? error : new Error(String(error)));
      return;
    }
    win.webContents.on("did-navigate", () => void checkCookie());
    win.webContents.on("did-redirect-navigation", () => void checkCookie());
    win.webContents.on("did-frame-navigate", () => void checkCookie());
    pollTimer = setInterval(() => void checkCookie(), 750);
    win.on("closed", () => {
      if (!settled) {
        finish(new Error("Sign-in window closed before authentication completed."));
      }
    });
    win.loadURL(portalBaseUrl).catch((error) => {
      finish(error instanceof Error ? error : new Error(String(error)));
    });
  });
}
async function discoverCloudAgents(org) {
  const portalBaseUrl = resolvePortalBaseUrl();
  if (!await hasLivePortalSession()) {
    const err = new Error(
      "You are not signed in to Hermes Cloud. Open Settings \u2192 Gateway, choose Hermes Cloud, and sign in."
    );
    err.needsCloudLogin = true;
    throw err;
  }
  const orgQuery = org ? `?org=${encodeURIComponent(org)}` : "";
  let body;
  try {
    body = await fetchJsonViaOauthSession(`${portalBaseUrl}/api/agents${orgQuery}`, {
      method: "GET",
      timeoutMs: 15e3
    });
  } catch (error) {
    if (error && error.statusCode === 401) {
      const err = new Error("Your Hermes Cloud session has expired. Open Settings \u2192 Gateway and sign in again.");
      err.needsCloudLogin = true;
      err.cause = error;
      throw err;
    }
    if (error && error.statusCode === 409) {
      const orgs = parseOrgSelectionError(error);
      if (orgs) {
        return { needsOrgSelection: true, orgs };
      }
    }
    throw error;
  }
  return { agents: trimCloudAgents(body), org: trimCloudOrg(body?.org) };
}
function trimCloudOrg(org) {
  if (!org || typeof org !== "object" || typeof org.id !== "string") {
    return null;
  }
  return {
    id: org.id,
    slug: typeof org.slug === "string" ? org.slug : null,
    name: typeof org.name === "string" ? org.name : org.id,
    isPersonal: Boolean(org.isPersonal),
    role: typeof org.role === "string" ? org.role : "MEMBER"
  };
}
function parseOrgSelectionError(error) {
  const msg = String(error?.message || "");
  const jsonStart = msg.indexOf("{");
  if (jsonStart < 0) {
    return null;
  }
  let parsed;
  try {
    parsed = JSON.parse(msg.slice(jsonStart));
  } catch {
    return null;
  }
  if (parsed?.error !== "org_selection_required" || !Array.isArray(parsed.orgs)) {
    return null;
  }
  return parsed.orgs.filter((o2) => o2 && typeof o2 === "object" && typeof o2.id === "string").map((o2) => ({
    id: o2.id,
    slug: typeof o2.slug === "string" ? o2.slug : null,
    name: typeof o2.name === "string" ? o2.name : o2.id,
    isPersonal: Boolean(o2.isPersonal),
    role: typeof o2.role === "string" ? o2.role : "MEMBER"
  }));
}
function trimCloudAgents(body) {
  const agents = Array.isArray(body?.agents) ? body.agents : [];
  return agents.filter((a) => a && typeof a === "object" && typeof a.id === "string").map((a) => ({
    id: a.id,
    name: typeof a.name === "string" ? a.name : a.id,
    status: typeof a.status === "string" ? a.status : "unknown",
    dashboardUrl: typeof a.dashboardUrl === "string" ? a.dashboardUrl : null,
    dashboardGatewayState: typeof a.dashboardGatewayState === "string" ? a.dashboardGatewayState : "unknown"
  }));
}
async function cloudAgentSilentSignIn(dashboardUrl) {
  const baseUrl = normalizeRemoteBaseUrl(dashboardUrl);
  if (!await hasLivePortalSession()) {
    const err = new Error("Your Hermes Cloud session has expired. Sign in to Hermes Cloud again.");
    err.needsCloudLogin = true;
    throw err;
  }
  await openOauthLoginWindow(baseUrl, { silent: true });
  return { baseUrl, connected: await hasOauthSessionCookie(baseUrl) };
}
function encryptDesktopSecret2(value) {
  return encryptDesktopSecret(value, safeStorage);
}
function decryptDesktopSecret(secret) {
  if (!secret || typeof secret !== "object") {
    return "";
  }
  const value = String(secret.value || "");
  if (!value) {
    return "";
  }
  if (secret.encoding === "safeStorage") {
    try {
      return safeStorage.decryptString(Buffer.from(value, "base64"));
    } catch {
      return "";
    }
  }
  return value;
}
function sanitizeConnectionProfiles(raw) {
  if (!raw || typeof raw !== "object") {
    return {};
  }
  const out = {};
  for (const [name, entry] of Object.entries(raw)) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    if (name !== "default" && !PROFILE_NAME_RE.test(name)) {
      continue;
    }
    if (entry.mode === "ssh") {
      const ssh = normalizeSshConfig(entry);
      if (ssh) {
        if (entry.token && typeof entry.token === "object") {
          ssh.token = entry.token;
        }
        out[name] = ssh;
      }
      continue;
    }
    const cleaned = {
      mode: modeIsRemoteLike(entry.mode) ? entry.mode : "local"
    };
    if (cleaned.mode === "local") {
      const savedSsh = normalizeSshConfig(entry.savedSsh);
      if (savedSsh) {
        cleaned.savedSsh = savedSsh;
      }
    }
    const url = String(entry.url || "").trim();
    if (url) {
      cleaned.url = url;
    }
    cleaned.authMode = normAuthMode(entry.authMode);
    if (entry.token && typeof entry.token === "object") {
      cleaned.token = entry.token;
    }
    if (cleaned.mode === "cloud") {
      const org = String(entry.org || "").trim();
      if (org) {
        cleaned.org = org;
      }
    }
    out[name] = cleaned;
  }
  return out;
}
function readDesktopConnectionConfig() {
  let mtime = null;
  try {
    mtime = fs19.statSync(DESKTOP_CONNECTION_CONFIG_PATH).mtimeMs;
  } catch {
    mtime = null;
  }
  if (connectionConfigCache && connectionConfigCacheMtime === mtime) {
    return connectionConfigCache;
  }
  let config = { mode: "local", remote: {}, profiles: {} };
  try {
    const raw = fs19.readFileSync(DESKTOP_CONNECTION_CONFIG_PATH, "utf8");
    const parsed = JSON.parse(raw);
    if (parsed && typeof parsed === "object") {
      const remote = parsed.remote && typeof parsed.remote === "object" ? parsed.remote : {};
      remote.authMode = remote.authMode === "oauth" ? "oauth" : "token";
      config = {
        mode: parsed.mode === "ssh" ? "ssh" : modeIsRemoteLike(parsed.mode) ? parsed.mode : "local",
        remote,
        // Per-profile remote overrides: each profile may point at its own
        // backend (local spawn or its own remote URL). Preserved verbatim so
        // profileRemoteOverride() can resolve them; normalized lazily on save.
        profiles: sanitizeConnectionProfiles(parsed.profiles)
      };
    }
  } catch {
  }
  connectionConfigCache = config;
  connectionConfigCacheMtime = mtime;
  return config;
}
function writeDesktopConnectionConfig(config) {
  fs19.mkdirSync(path22.dirname(DESKTOP_CONNECTION_CONFIG_PATH), { recursive: true });
  writeFileAtomic(DESKTOP_CONNECTION_CONFIG_PATH, JSON.stringify(config, null, 2));
  connectionConfigCache = config;
  connectionConfigCacheMtime = fs19.statSync(DESKTOP_CONNECTION_CONFIG_PATH).mtimeMs;
}
function readActiveDesktopProfile() {
  try {
    const raw = fs19.readFileSync(DESKTOP_PROFILE_CONFIG_PATH, "utf8");
    const parsed = JSON.parse(raw);
    const name = parsed && typeof parsed.profile === "string" ? parsed.profile.trim() : "";
    if (name && (name === "default" || PROFILE_NAME_RE.test(name))) {
      return name;
    }
  } catch {
  }
  return null;
}
function writeActiveDesktopProfile(name) {
  const value = typeof name === "string" ? name.trim() : "";
  if (value && value !== "default" && !PROFILE_NAME_RE.test(value)) {
    throw new Error(`Invalid profile name: ${value}`);
  }
  fs19.mkdirSync(path22.dirname(DESKTOP_PROFILE_CONFIG_PATH), { recursive: true });
  writeFileAtomic(DESKTOP_PROFILE_CONFIG_PATH, JSON.stringify({ profile: value || null }, null, 2));
  return value || null;
}
async function sanitizeDesktopConnectionConfig(config = readDesktopConnectionConfig(), profile = null) {
  const key = connectionScopeKey(profile);
  const scoped = key ? config.profiles?.[key] || null : null;
  const block = key ? scoped || {} : config.remote || {};
  const envOverride = key ? false : Boolean(process.env.HERMES_DESKTOP_REMOTE_URL);
  const savedMode = key ? scoped?.mode : config.mode;
  const ssh = savedMode === "ssh" ? normalizeSshConfig(block) : null;
  const savedSsh = savedMode === "local" ? key ? savedProfileSsh(config, key) : normalizeSshConfig(block) : null;
  const remoteToken = decryptDesktopSecret(block.token);
  const authMode = normAuthMode(block.authMode);
  const remoteUrl = envOverride ? String(process.env.HERMES_DESKTOP_REMOTE_URL || "") : String(block.url || "");
  const mode = envOverride ? "remote" : savedMode === "ssh" ? "ssh" : modeIsRemoteLike(savedMode) ? savedMode : "local";
  let remoteOauthConnected = false;
  if (authMode === "oauth" && remoteUrl) {
    try {
      remoteOauthConnected = oauthSessionIsLive(hasNativeSession(remoteUrl), await hasLiveOauthSession(remoteUrl));
    } catch {
      remoteOauthConnected = false;
    }
  }
  return {
    mode,
    // Echo the scope back so the UI knows which profile (if any) this reflects.
    profile: key,
    remoteAuthMode: authMode,
    remoteOauthConnected,
    remoteUrl,
    // The persisted Hermes Cloud org (slug/id) for a cloud connection, or '' for
    // remote/local. Lets Settings → Gateway reopen into the same org.
    cloudOrg: mode === "cloud" ? String(block.org || "") : "",
    remoteTokenPreview: tokenPreview(remoteToken),
    remoteTokenSet: Boolean(remoteToken),
    sshHost: (ssh || savedSsh)?.host || "",
    sshUser: (ssh || savedSsh)?.user || "",
    sshPort: (ssh || savedSsh)?.port || null,
    sshKeyPath: (ssh || savedSsh)?.keyPath || "",
    sshRemoteHermesPath: (ssh || savedSsh)?.remoteHermesPath || "",
    sshRemoteProfile: (ssh || savedSsh)?.remoteProfile || "",
    // The env override only forces the global/primary connection; a per-profile
    // scope is never overridden by HERMES_DESKTOP_REMOTE_URL.
    envOverride
  };
}
function buildRemoteBlock(remoteUrl, authMode, token, org) {
  if (authMode !== "oauth" && !decryptDesktopSecret(token)) {
    throw new Error("Remote gateway session token is required.");
  }
  const block = {
    url: normalizeRemoteBaseUrl(remoteUrl),
    authMode,
    token
  };
  const orgValue = typeof org === "string" ? org.trim() : "";
  if (orgValue) {
    block.org = orgValue;
  }
  return block;
}
function coerceDesktopConnectionConfig(input = {}, existing = readDesktopConnectionConfig(), options = {}) {
  const persistToken = options.persistToken !== false;
  const key = connectionScopeKey(input.profile);
  const mode = input.mode === "ssh" ? "ssh" : modeIsRemoteLike(input.mode) ? input.mode : "local";
  const remoteLike = modeIsRemoteLike(mode);
  const rawExistingBlock = key ? existing.profiles?.[key] || {} : existing.remote || {};
  const existingMode = key ? existing.profiles?.[key]?.mode : existing.mode;
  const leavingCloud = existingMode === "cloud" && mode !== "cloud";
  const leavingSsh = rawExistingBlock.mode === "ssh" && mode !== "ssh" && mode !== "local";
  const existingBlock = leavingCloud || leavingSsh ? {} : rawExistingBlock;
  const remoteUrl = String(input.remoteUrl ?? existingBlock.url ?? "").trim();
  const authMode = resolveAuthMode(input.remoteAuthMode, existingBlock.authMode);
  const cloudOrg = mode === "cloud" ? String(input.cloudOrg ?? existingBlock.org ?? "").trim() : "";
  const incomingToken = typeof input.remoteToken === "string" ? input.remoteToken.trim() : "";
  const nextToken = incomingToken ? persistToken ? encryptDesktopSecret2(incomingToken) : { encoding: "plain", value: incomingToken } : existingBlock.token;
  if (mode === "ssh") {
    const sshBlock = buildSshBlock(input, savedProfileSsh(existing, key) || rawExistingBlock);
    if (key) {
      const profiles = { ...existing.profiles || {}, [key]: sshBlock };
      return {
        mode: existing.mode === "ssh" || modeIsRemoteLike(existing.mode) ? existing.mode : "local",
        remote: existing.remote || {},
        profiles
      };
    }
    return { mode: "ssh", remote: sshBlock, profiles: existing.profiles || {} };
  }
  if (key) {
    const profiles = { ...existing.profiles || {} };
    if (remoteLike) {
      profiles[key] = { mode, ...buildRemoteBlock(remoteUrl, authMode, nextToken, cloudOrg) };
    } else {
      const localEntry = localProfileEntry(rawExistingBlock);
      if (localEntry) {
        profiles[key] = localEntry;
      } else {
        delete profiles[key];
      }
    }
    return {
      mode: existing.mode === "ssh" || modeIsRemoteLike(existing.mode) ? existing.mode : "local",
      remote: existing.remote || {},
      profiles
    };
  }
  const nextRemote = remoteLike ? buildRemoteBlock(remoteUrl, authMode, nextToken, cloudOrg) : existingMode === "ssh" ? rawExistingBlock : { url: remoteUrl ? normalizeRemoteBaseUrl(remoteUrl) : remoteUrl, authMode, token: nextToken };
  return { mode, remote: nextRemote, profiles: existing.profiles || {} };
}
function buildSshBlock(input, existingBlock = {}) {
  const merged = normalizeSshConfig({
    mode: "ssh",
    host: input.sshHost ?? existingBlock.host,
    user: input.sshUser ?? existingBlock.user,
    port: input.sshPort ?? existingBlock.port,
    keyPath: input.sshKeyPath ?? existingBlock.keyPath,
    remoteHermesPath: input.sshRemoteHermesPath ?? existingBlock.remoteHermesPath,
    remoteProfile: input.sshRemoteProfile ?? existingBlock.remoteProfile
  });
  if (!merged) {
    throw new Error("SSH host is required.");
  }
  if (existingBlock.token && existingBlock.host === merged.host) {
    merged.token = existingBlock.token;
  }
  return merged;
}
async function buildRemoteConnection(rawUrl, authMode, token, source, remoteHost, remoteKind = "url", remoteIdentity) {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl);
  const host = remoteHost || hostLabelFromBaseUrl(baseUrl);
  if (authMode === "oauth") {
    if (!oauthSessionIsLive(hasNativeSession(baseUrl), await hasLiveOauthSession(baseUrl)) && oauthGuardMayHardFail(await gatewayAuthProviders(baseUrl))) {
      const err = new Error(
        'Remote Hermes gateway uses OAuth, but you are not signed in. Open Settings \u2192 Gateway and click "Sign in", or switch back to Local.'
      );
      err.needsOauthLogin = true;
      throw err;
    }
    let ticket;
    try {
      ticket = await mintGatewayWsTicket(baseUrl);
    } catch (error) {
      throw gatewayTicketFailure(
        error,
        'Your remote gateway session has expired. Open Settings \u2192 Gateway and click "Sign in" again.',
        "Could not reach the remote Hermes gateway while refreshing its WebSocket ticket. Try reconnecting."
      );
    }
    return {
      baseUrl,
      mode: "remote",
      source,
      authMode: "oauth",
      remoteHost: host || void 0,
      remoteIdentity,
      remoteKind,
      // No static token in OAuth mode; REST is cookie-authed via the partition.
      token: null,
      wsUrl: buildGatewayWsUrlWithTicket(baseUrl, ticket)
    };
  }
  if (!token) {
    throw new Error(
      "Remote Hermes gateway is selected, but no session token is saved. Open Settings \u2192 Gateway and save a token, or switch back to Local."
    );
  }
  return {
    baseUrl,
    mode: "remote",
    source,
    authMode: "token",
    remoteHost: host || void 0,
    remoteIdentity,
    remoteKind,
    token,
    wsUrl: buildGatewayWsUrl(baseUrl, token)
  };
}
var sshConnections = /* @__PURE__ */ new Map();
var desktopInstallationId = loadOrCreateInstallationId(DESKTOP_INSTALLATION_PATH);
var sshBootstrapCoordinator = createBootstrapCoordinator();
var sshQuitTeardownDone = false;
function sshScopeKey(profile) {
  return connectionScopeKey(profile) || "";
}
function sshOwnershipKey(profile) {
  return sshOwnershipId(desktopInstallationId, sshScopeKey(profile));
}
function sshRememberLog(chunk) {
  rememberLog(redactSecrets(String(chunk == null ? "" : chunk)));
}
async function sshProbeReuseProof(baseUrl, token, spawnNonce) {
  try {
    const proof = await fetchJson(`${baseUrl}/api/ssh/ownership`, token);
    return proof?.ok === true && proof.sshOwnerNonce === spawnNonce && proof.protocolVersion === 1 ? "authenticated-ok" : "authenticated-stale";
  } catch (error) {
    if (/^(401|403|404):/.test(String(error?.message || ""))) {
      return "authenticated-stale";
    }
    throw error;
  }
}
async function teardownSshConnection(profile) {
  const scope = sshScopeKey(profile);
  const state = sshConnections.get(scope);
  if (!state) {
    return;
  }
  sshConnections.delete(scope);
  for (const [id, info] of [...terminalSessions.entries()]) {
    if (info.sshScope === scope) {
      disposeTerminalSession(id);
    }
  }
  try {
    if (state.localPort && state.remotePort) {
      await state.ssh.cancelForward(state.localPort, state.remotePort);
    }
  } catch {
  }
  try {
    await state.ssh.close();
  } catch {
  }
}
function activeSshTerminalTarget() {
  const profile = primaryProfileKey();
  const config = readDesktopConnectionConfig();
  if (profileSshOverride(config, profile)) {
    const scope = sshScopeKey(profile);
    const state = sshConnections.get(scope);
    return state && state.ssh ? { ssh: state.ssh, scope } : "pending";
  }
  if (profileRemoteOverride(config, profile)) {
    return null;
  }
  if (process.env.HERMES_DESKTOP_REMOTE_URL) {
    return null;
  }
  if (config.mode === "ssh") {
    const state = sshConnections.get("");
    return state && state.ssh ? { ssh: state.ssh, scope: "" } : "pending";
  }
  return null;
}
function effectiveSshConfigFingerprint(sshConfig) {
  const ssh = process.platform === "win32" ? path22.join(process.env.SystemRoot || "C:\\Windows", "System32", "OpenSSH", "ssh.exe") : "ssh";
  const args = ["-G"];
  if (sshConfig.port) {
    args.push("-p", String(sshConfig.port));
  }
  if (sshConfig.keyPath) {
    args.push("-i", sshConfig.keyPath);
  }
  args.push("--", sshConfig.user ? `${sshConfig.user}@${sshConfig.host}` : sshConfig.host);
  const output = execFileSync6(ssh, args, { encoding: "utf8", timeout: 1e4, windowsHide: true });
  return crypto6.createHash("sha256").update(output).digest("hex");
}
async function bootstrapSshConnection(profile, sshConfig, reuseToken, source) {
  const scope = sshScopeKey(profile);
  const effectiveConfigFingerprint = effectiveSshConfigFingerprint(sshConfig);
  const resolvedConfig = { ...sshConfig, effectiveConfigFingerprint };
  const fingerprint = sshConfigFingerprint(scope, resolvedConfig);
  return sshBootstrapCoordinator.start(
    scope,
    fingerprint,
    (lease) => bootstrapSshConnectionInner(profile, resolvedConfig, reuseToken, source, fingerprint, lease)
  );
}
async function bootstrapSshConnectionInner(profile, sshConfig, reuseToken, source, fingerprint, lease) {
  const scope = sshScopeKey(profile);
  const hostLabel = sshConfig.user ? `${sshConfig.user}@${sshConfig.host}` : sshConfig.host;
  const existing = sshConnections.get(scope);
  if (existing && existing.fingerprint !== fingerprint) {
    await teardownSshConnection(profile);
  }
  let ssh = sshConnections.get(scope)?.ssh;
  if (ssh && !await ssh.isAlive()) {
    try {
      await ssh.close();
    } catch {
    }
    ssh = null;
    sshConnections.delete(scope);
  }
  const created = !ssh;
  let removeForceCleanup = () => {
  };
  if (created) {
    ssh = new SshConnection(
      { host: sshConfig.host, user: sshConfig.user, port: sshConfig.port, keyPath: sshConfig.keyPath },
      {
        rememberLog: sshRememberLog,
        ownershipId: sshOwnershipKey(profile),
        scope,
        effectiveConfigFingerprint: sshConfig.effectiveConfigFingerprint
      }
    );
    removeForceCleanup = lease.onForceCleanup(() => ssh.close());
    await ssh.open();
  }
  let result;
  try {
    const platform = await detectRemotePlatform(ssh, sshConfig.remoteHermesPath || "");
    const lifecycle = platform.os === "Windows" ? connectWindowsRemote : connect;
    result = await lifecycle({
      ssh,
      profile: sshConfig.remoteProfile || connectionScopeKey(profile) || "",
      remoteHermesPath: sshConfig.remoteHermesPath || "",
      ownershipId: sshOwnershipKey(profile),
      reuseToken: reuseToken || "",
      forward: (localPort, remotePort) => ssh.forward(localPort, remotePort),
      cancelForward: (localPort, remotePort) => ssh.cancelForward(localPort, remotePort),
      pickLocalPort,
      waitForHermes: (baseUrl, token) => waitForHermes(baseUrl, token, lease.signal, "token"),
      probeReuseProof: sshProbeReuseProof,
      adoptServedToken: adoptServedDashboardToken,
      rememberLog: sshRememberLog,
      signal: lease.signal
    });
  } catch (error) {
    if (created) {
      try {
        await ssh.close();
      } catch {
      }
    }
    const err = new Error(error.message);
    err.sshError = error.kind || "unknown";
    err.isSshBootstrap = true;
    throw err;
  }
  try {
    lease.assertCurrent();
  } catch (error) {
    try {
      await ssh.cancelForward(result.localPort, result.remotePort);
      await ssh.close();
    } catch {
    }
    throw error;
  }
  persistSshConnectionToken(profile, source, result.token);
  removeForceCleanup();
  sshConnections.set(scope, {
    ssh,
    fingerprint,
    localPort: result.localPort,
    remotePort: result.remotePort,
    pid: result.pid,
    host: sshConfig.host,
    hostLabel,
    hermesVersion: result.hermesVersion || "",
    remotePlatform: result.platform?.os || "",
    reused: result.reused
  });
  sshRememberLog(
    `[ssh] connection ${result.reused ? "REUSED" : "spawned"} dashboard: ${result.hermesVersion || "hermes (version unknown)"} at ${result.hermesPath || "?"}`
  );
  const connection = await buildRemoteConnection(
    result.baseUrl,
    "token",
    result.token,
    source,
    hostLabel,
    "ssh",
    result.ownershipId
  );
  return { ...connection, remoteHermesVersion: result.hermesVersion || "" };
}
function persistSshConnectionToken(profile, source, token) {
  try {
    const config = readDesktopConnectionConfig();
    const encrypted = encryptDesktopSecret2(token);
    if (source === "profile") {
      const key = connectionScopeKey(profile);
      if (key && config.profiles?.[key]?.mode === "ssh") {
        config.profiles[key].token = encrypted;
        writeDesktopConnectionConfig(config);
      }
    } else if (config.mode === "ssh" && config.remote) {
      config.remote.token = encrypted;
      writeDesktopConnectionConfig(config);
    }
  } catch (error) {
    sshRememberLog(`[ssh] could not persist served token: ${error.message}`);
  }
}
async function resolveRemoteBackend(profile) {
  const config = readDesktopConnectionConfig();
  const sshOverride = profileSshOverride(config, profile);
  if (sshOverride) {
    const reuseToken = decryptDesktopSecret(config.profiles?.[connectionScopeKey(profile)]?.token);
    return bootstrapSshConnection(profile, sshOverride, reuseToken, "profile");
  }
  const override = profileRemoteOverride(config, profile);
  if (override) {
    const token2 = override.authMode === "oauth" ? null : decryptDesktopSecret(override.token);
    return buildRemoteConnection(
      override.url,
      override.authMode,
      token2,
      "profile",
      void 0,
      config.profiles?.[connectionScopeKey(profile)]?.mode === "cloud" ? "cloud" : "url"
    );
  }
  const rawEnvUrl = process.env.HERMES_DESKTOP_REMOTE_URL;
  const rawEnvToken = process.env.HERMES_DESKTOP_REMOTE_TOKEN;
  if (rawEnvUrl) {
    if (!rawEnvToken) {
      throw new Error(
        "HERMES_DESKTOP_REMOTE_URL is set but HERMES_DESKTOP_REMOTE_TOKEN is not. Both must be provided to connect to a remote Hermes backend."
      );
    }
    return buildRemoteConnection(rawEnvUrl, "token", rawEnvToken, "env");
  }
  if (config.mode === "ssh") {
    const ssh = normalizeSshConfig({ mode: "ssh", ...config.remote || {} });
    if (!ssh) {
      throw new Error("SSH remote mode is selected but no host is configured.");
    }
    const reuseToken = decryptDesktopSecret(config.remote?.token);
    return bootstrapSshConnection(null, ssh, reuseToken, "settings");
  }
  if (!modeIsRemoteLike(config.mode)) {
    return null;
  }
  const authMode = normAuthMode(config.remote?.authMode);
  const token = authMode === "oauth" ? null : decryptDesktopSecret(config.remote?.token);
  return buildRemoteConnection(
    config.remote?.url,
    authMode,
    token,
    "settings",
    void 0,
    config.mode === "cloud" ? "cloud" : "url"
  );
}
function profileHasRemoteOverride(profile) {
  return profileHasRemoteConnection(readDesktopConnectionConfig(), profile);
}
function configuredRemoteProfileNames() {
  const config = readDesktopConnectionConfig();
  return Object.keys(config.profiles || {}).filter((name) => profileHasRemoteConnection(config, name));
}
function globalRemoteActive() {
  if (process.env.HERMES_DESKTOP_REMOTE_URL) {
    return true;
  }
  const mode = readDesktopConnectionConfig().mode;
  return modeIsRemoteLike(mode) || mode === "ssh";
}
function primaryBackendIsRemote() {
  return Boolean(profileHasRemoteOverride(primaryProfileKey())) || globalRemoteActive();
}
async function fetchJsonForProfile(profile, path23) {
  return requestJsonForProfile(profile, path23, "GET");
}
async function requestJsonForProfile(profile, path23, method, body) {
  const conn = await ensureBackend(profile);
  const url = `${conn.baseUrl}${path23}`;
  const opts = { method, body, timeoutMs: DEFAULT_FETCH_TIMEOUT_MS };
  if (conn.authMode === "oauth") {
    const nativeAt = await ensureNativeAccessToken(conn.baseUrl).catch(() => null);
    if (nativeAt) {
      return fetchJson(url, null, { ...opts, bearer: nativeAt });
    }
    return fetchJsonViaOauthSession(url, opts);
  }
  return fetchJson(url, conn.token, opts);
}
async function probeRemoteAuthMode(rawUrl) {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl);
  let status;
  try {
    status = await fetchPublicJson(`${baseUrl}/api/status`, { timeoutMs: 8e3 });
  } catch (error) {
    return {
      baseUrl,
      reachable: false,
      authMode: "unknown",
      providers: [],
      version: null,
      error: error instanceof Error ? error.message : String(error)
    };
  }
  const authRequired = authModeFromStatus(status) === "oauth";
  let providers = [];
  if (authRequired) {
    try {
      const body = await fetchPublicJson(`${baseUrl}/api/auth/providers`, { timeoutMs: 8e3 });
      if (Array.isArray(body?.providers)) {
        providers = body.providers.filter((p2) => p2 && typeof p2 === "object").map((p2) => ({
          name: String(p2.name || ""),
          displayName: String(p2.display_name || p2.name || ""),
          supportsPassword: Boolean(p2.supports_password)
        })).filter((p2) => p2.name);
      }
    } catch {
    }
  }
  return {
    baseUrl,
    reachable: true,
    authMode: authRequired ? "oauth" : "token",
    providers,
    version: status?.version || null,
    error: null
  };
}
async function testDesktopConnectionConfig(input = {}) {
  if (input.mode === "ssh") {
    const sshConfig = normalizeSshConfig({
      mode: "ssh",
      host: input.sshHost,
      user: input.sshUser,
      port: input.sshPort,
      keyPath: input.sshKeyPath,
      remoteHermesPath: input.sshRemoteHermesPath
    });
    if (!sshConfig) {
      return { reachable: false, sshError: "unreachable", error: "SSH host is required." };
    }
    const ssh = createSshProbeConnection(
      { host: sshConfig.host, user: sshConfig.user, port: sshConfig.port, keyPath: sshConfig.keyPath },
      { rememberLog: sshRememberLog }
    );
    try {
      let attempt = 0;
      for (; ; ) {
        try {
          await ssh.open();
          const platform = await detectRemotePlatform(ssh, sshConfig.remoteHermesPath || "");
          let hermesPath;
          let hermesVersion;
          let supported;
          if (platform.os === "Windows") {
            const runtime = platform;
            hermesPath = runtime.hermesPath;
            const inspection = await helper(ssh, runtime, "inspect", [runtime.hermesPath]);
            hermesVersion = inspection.version;
            supported = inspection.supported;
          } else {
            hermesPath = await locateHermes(ssh, sshConfig.remoteHermesPath || "");
            hermesVersion = await probeHermesVersion(ssh, hermesPath);
            supported = await remoteSupportsSshOwnership(ssh, hermesPath);
          }
          if (!supported) {
            return {
              reachable: false,
              sshError: "update-required",
              error: "Update Hermes on the remote host before connecting with Desktop SSH."
            };
          }
          return {
            reachable: true,
            sshError: null,
            error: null,
            remotePlatform: `${platform.os}/${platform.arch}`,
            remoteHermesPath: hermesPath,
            remoteHermesVersion: hermesVersion,
            host: sshConfig.user ? `${sshConfig.user}@${sshConfig.host}` : sshConfig.host
          };
        } catch (error) {
          if (error?.kind === "timeout" && attempt === 0) {
            attempt += 1;
            sshRememberLog("[ssh] test probe timed out once; retrying");
            continue;
          }
          throw error;
        }
      }
    } catch (error) {
      return { reachable: false, sshError: error.kind || "unknown", error: error.message };
    } finally {
      try {
        await ssh.close();
      } catch {
      }
    }
  }
  const config = coerceDesktopConnectionConfig(input, readDesktopConnectionConfig(), { persistToken: false });
  const key = connectionScopeKey(input.profile);
  const block = key ? config.profiles?.[key] || null : config.remote;
  const wantRemote = modeIsRemoteLike(block?.mode) || !key && modeIsRemoteLike(config.mode) || modeIsRemoteLike(input.mode) && block;
  let baseUrl;
  let token = null;
  let authMode = "token";
  if (wantRemote && block?.url) {
    baseUrl = normalizeRemoteBaseUrl(block.url);
    authMode = normAuthMode(block.authMode);
    if (authMode !== "oauth") {
      token = decryptDesktopSecret(block.token);
    }
  } else {
    const remote = await resolveRemoteBackend(key) || await startHermes();
    baseUrl = remote.baseUrl;
    token = remote.token;
    authMode = normAuthMode(remote.authMode);
  }
  const status = await fetchJson(`${baseUrl}/api/status`, token, { timeoutMs: 8e3 });
  const wsUrl = await resolveTestWsUrl(baseUrl, authMode, token, { mintTicket: mintGatewayWsTicket });
  if (wsUrl && typeof globalThis.WebSocket === "function") {
    const probe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket });
    if (!probe.ok) {
      throw new Error(
        `Reached the gateway over HTTP, but the live WebSocket (/api/ws) connection failed: ${probe.reason} The HTTP check can pass while the WebSocket is blocked by a proxy, firewall, or gateway auth/origin guard.`
      );
    }
  }
  return {
    ok: true,
    baseUrl,
    version: status?.version || null
  };
}
function resetBootProgressForReconnect() {
  updateBootProgress(
    {
      error: null,
      message: "Restarting desktop connection",
      phase: "backend.resolve",
      progress: 4,
      running: true
    },
    { allowDecrease: true }
  );
}
function stopBackendChild2(child) {
  stopBackendChild(child, { forceKillProcessTree, isWindows: IS_WINDOWS3 });
}
function resetHermesConnection({ soft = false } = {}) {
  backendStartFailure = null;
  remoteReauthFailure = null;
  remoteLiveness.clear();
  const hermesProcess = backendConnectionState.invalidate();
  stopBackendChild2(hermesProcess);
  if (!soft) {
    resetBootProgressForReconnect();
  }
}
async function teardownPrimaryBackendAndWait({ soft = false } = {}) {
  const hermesProcess = backendConnectionState.getProcess();
  const dying = hermesProcess && !hermesProcess.killed ? hermesProcess : null;
  if (soft) {
    softRehomeInProgress = true;
  }
  try {
    resetHermesConnection({ soft });
    await waitForBackendExit(dying);
  } finally {
    if (soft) {
      softRehomeInProgress = false;
    }
  }
}
function sendConnectionApplied() {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  const { webContents } = mainWindow;
  if (!webContents || webContents.isDestroyed()) {
    return;
  }
  webContents.send("hermes:connection:applied");
}
async function waitForBackendExit(child, timeoutMs = 5e3) {
  if (!child) {
    return;
  }
  if (child.exitCode !== null || child.signalCode !== null) {
    return;
  }
  await new Promise((resolve) => {
    const timer = setTimeout(() => {
      try {
        if (IS_WINDOWS3 && Number.isInteger(child.pid)) {
          forceKillProcessTree(child.pid);
        } else {
          child.kill("SIGKILL");
        }
      } catch {
      }
      resolve();
    }, timeoutMs);
    child.once("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}
function primaryProfileKey() {
  return readActiveDesktopProfile() || "default";
}
function profileRouteOptions(profile) {
  return {
    globalRemote: globalRemoteActive(),
    primaryProfile: primaryProfileKey(),
    profileRemoteOverride: Boolean(profileHasRemoteOverride(profile))
  };
}
async function ensureBackend(profile) {
  const key = profile && String(profile).trim() ? String(profile).trim() : primaryProfileKey();
  const route = resolveProfileBackendRoute(key, profileRouteOptions(key));
  if (route.backend === "primary") {
    const connection = await startHermes();
    return route.descriptorProfile ? { ...connection, profile: route.descriptorProfile } : connection;
  }
  const existing = backendPool.get(key);
  if (existing) {
    existing.lastActiveAt = Date.now();
    return existing.connectionPromise;
  }
  evictLruPoolBackends(POOL_MAX_BACKENDS - 1);
  const entry = {
    process: null,
    port: null,
    token: null,
    connectionPromise: null,
    lastActiveAt: Date.now(),
    remoteBaseUrl: null
  };
  entry.connectionPromise = spawnPoolBackend(key, entry).catch((error) => {
    backendPool.delete(key);
    throw error;
  });
  backendPool.set(key, entry);
  startPoolIdleReaper();
  return entry.connectionPromise;
}
function touchPoolBackend(profile) {
  const key = profile && String(profile).trim() ? String(profile).trim() : null;
  if (!key) {
    return;
  }
  const entry = backendPool.get(key);
  if (entry) {
    entry.lastActiveAt = Date.now();
  }
}
function evictLruPoolBackends(keep) {
  if (backendPool.size <= keep) {
    return;
  }
  const now = Date.now();
  const evictable = [...backendPool.entries()].filter(([, entry]) => now - (entry.lastActiveAt || 0) > POOL_KEEPALIVE_FRESH_MS).sort((a, b2) => (a[1].lastActiveAt || 0) - (b2[1].lastActiveAt || 0));
  let removable = backendPool.size - Math.max(0, keep);
  for (const [profile] of evictable) {
    if (removable <= 0) {
      break;
    }
    rememberLog(`Evicting idle profile backend "${profile}" (LRU cap ${POOL_MAX_BACKENDS})`);
    stopPoolBackend(profile);
    removable -= 1;
  }
}
function startPoolIdleReaper() {
  if (poolIdleReaper) {
    return;
  }
  poolIdleReaper = setInterval(() => {
    const now = Date.now();
    for (const [profile, entry] of [...backendPool.entries()]) {
      if (now - (entry.lastActiveAt || 0) > POOL_IDLE_MS) {
        rememberLog(`Reaping idle profile backend "${profile}" (idle > ${Math.round(POOL_IDLE_MS / 1e3)}s)`);
        stopPoolBackend(profile);
      }
    }
    if (backendPool.size === 0 && poolIdleReaper) {
      clearInterval(poolIdleReaper);
      poolIdleReaper = null;
    }
  }, 6e4);
  if (typeof poolIdleReaper.unref === "function") {
    poolIdleReaper.unref();
  }
}
async function spawnPoolBackend(profile, entry) {
  const remote = await resolveRemoteBackend(profile);
  if (remote) {
    await waitForHermes(remote.baseUrl, remote.token, void 0, remote.authMode);
    entry.remoteBaseUrl = remote.baseUrl;
    return {
      ...remote,
      profile,
      logs: hermesLog.slice(-80),
      ...getWindowState()
    };
  }
  const token = crypto6.randomBytes(32).toString("base64url");
  {
    let poolAnnounced = false;
    await waitForUpdateClearance(updateGateDeps(), {
      onWaitTick: (reason) => {
        if (!poolAnnounced) {
          poolAnnounced = true;
          rememberLog(`[updates] update in progress (${reason}); deferring pool backend start for profile "${profile}"`);
        }
      },
      pollMs: UPDATE_WAIT_POLL_MS,
      timeoutMs: UPDATE_WAIT_TIMEOUT_MS
    });
  }
  const backendArgs = ["--profile", profile, "serve", "--host", "127.0.0.1", "--port", "0"];
  const backend = await ensureRuntime(resolveHermesBackend(backendArgs));
  backend.args = getBackendArgsForRuntime(backend);
  const hermesCwd = resolveHermesCwd();
  const webDist = resolveWebDist();
  const readyFile = backend.readyFile ? makeDashboardReadyFile() : null;
  rememberLog(`Starting Hermes backend for profile "${profile}" via ${backend.label}`);
  const child = spawn5(
    backend.command,
    backend.args,
    hiddenWindowsChildOptions({
      cwd: hermesCwd,
      env: {
        ...process.env,
        HERMES_HOME,
        ...backend.env,
        // Pin the gateway's tool/terminal cwd to the same directory we chose for
        // the child process. Inherited TERMINAL_CWD (or a stale config bridge)
        // can still point at the install dir even when spawn cwd is home.
        TERMINAL_CWD: hermesCwd,
        HERMES_DASHBOARD_SESSION_TOKEN: token,
        // Marks this dashboard backend as desktop-spawned so it runs the cron
        // scheduler tick loop (the gateway isn't running under the app).
        HERMES_DESKTOP: "1",
        HERMES_WEB_DIST: webDist,
        ...readyFile ? { HERMES_DESKTOP_READY_FILE: readyFile } : {}
      },
      shell: backend.shell,
      stdio: ["ignore", "pipe", "pipe"]
    })
  );
  entry.process = child;
  entry.token = token;
  child.stdout.on("data", rememberLog);
  child.stderr.on("data", rememberLog);
  let ready = false;
  let rejectStart = null;
  const startFailed = new Promise((_resolve, reject) => {
    rejectStart = reject;
  });
  child.once("error", (error) => {
    rememberLog(`Hermes backend for profile "${profile}" failed to start: ${error.message}`);
    backendPool.delete(profile);
    rejectStart?.(error);
  });
  child.once("exit", (code, signal) => {
    rememberLog(`Hermes backend for profile "${profile}" exited (${signal || code})`);
    backendPool.delete(profile);
    if (!ready) {
      rejectStart?.(
        new Error(`Hermes backend for profile "${profile}" exited before it became ready (${signal || code}).`)
      );
    }
  });
  const port = await Promise.race([waitForDashboardPortAnnouncement(child, { readyFile }), startFailed]);
  if (readyFile) {
    fs19.unlink(readyFile, () => {
    });
  }
  entry.port = port;
  const baseUrl = `http://127.0.0.1:${port}`;
  await Promise.race([waitForHermes(baseUrl, token), startFailed]);
  ready = true;
  const authToken = await adoptServedDashboardToken(baseUrl, token, {
    childAlive: () => child.exitCode === null && !child.killed,
    label: `Hermes backend for profile "${profile}"`,
    rememberLog
  });
  entry.token = authToken;
  const wsUrl = `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`;
  const wsProbe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket });
  if (!wsProbe.ok) {
    throw new Error(
      `Hermes backend for profile "${profile}" is HTTP-reachable but the WebSocket (/api/ws) rejected the session token: ${wsProbe.reason}`
    );
  }
  return {
    baseUrl,
    mode: "local",
    source: "local",
    authMode: "token",
    token: authToken,
    profile,
    wsUrl,
    logs: hermesLog.slice(-80),
    ...getWindowState()
  };
}
function stopPoolBackend(profile) {
  const entry = backendPool.get(profile);
  if (!entry) {
    return;
  }
  backendPool.delete(profile);
  stopBackendChild2(entry.process);
}
async function teardownPoolBackendAndWait(profile) {
  const entry = backendPool.get(profile);
  if (!entry) {
    return;
  }
  backendPool.delete(profile);
  stopBackendChild2(entry.process);
  await waitForBackendExit(entry.process);
}
function stopAllPoolBackends() {
  for (const profile of [...backendPool.keys()]) {
    stopPoolBackend(profile);
  }
}
async function prepareProfileDeleteRequest(request2) {
  const profile = profileNameFromDeleteRequest(request2);
  const decision = decideProfileDeleteAction(profile, {
    isDefaultProfile: (p2) => p2 === "default",
    isValidProfileName: (p2) => PROFILE_NAME_RE.test(p2),
    primaryProfileKey
  });
  if (decision.action === "noop") {
    return null;
  }
  if (decision.action === "teardown-primary") {
    writeActiveDesktopProfile("default");
    await teardownPrimaryBackendAndWait();
    return decision.profile;
  }
  await teardownPoolBackendAndWait(decision.profile);
  return decision.profile;
}
async function startHermes() {
  if (bootstrapFailure) {
    throw bootstrapFailure;
  }
  if (backendStartFailure) {
    throw backendStartFailure;
  }
  if (remoteReauthFailure) {
    throw remoteReauthFailure;
  }
  if (BOOT_FAKE_ERROR) {
    await advanceBootProgress("backend.resolve", "Resolving Hermes backend", 8);
    const error = new Error(BOOT_FAKE_ERROR);
    error.isBootstrapFailure = true;
    bootstrapFailure = error;
    throw error;
  }
  const existingConnectionPromise = backendConnectionState.getPromise();
  if (existingConnectionPromise) {
    return existingConnectionPromise;
  }
  const connectionAttempt = backendConnectionState.startAttempt();
  let attemptedRemote = primaryBackendIsRemote();
  const connectionPromise = (async () => {
    const connectRemote = async (remote) => {
      await advanceBootProgress("backend.remote", `Connecting to remote Hermes backend at ${remote.baseUrl}`, 24);
      await waitForHermes(remote.baseUrl, remote.token, void 0, remote.authMode);
      updateBootProgress({
        phase: "backend.ready",
        message: "Remote Hermes backend is ready",
        progress: 94,
        running: true,
        error: null
      });
      return {
        baseUrl: remote.baseUrl,
        mode: "remote",
        source: remote.source,
        authMode: remote.authMode || "token",
        remoteHost: remote.remoteHost,
        remoteKind: remote.remoteKind,
        remoteHermesVersion: remote.remoteHermesVersion,
        token: remote.token,
        wsUrl: remote.wsUrl,
        logs: hermesLog.slice(-80),
        ...getWindowState()
      };
    };
    await advanceBootProgress("backend.resolve", "Resolving Hermes backend", 8);
    const token = crypto6.randomBytes(32).toString("base64url");
    const backendArgs = ["serve", "--host", "127.0.0.1", "--port", "0"];
    const activeProfile = readActiveDesktopProfile();
    if (activeProfile) {
      backendArgs.unshift("--profile", activeProfile);
    }
    const setup = await runPrimaryBackendStartup({
      connectRemote,
      ensureLocalRuntime: ensureRuntime,
      prepareLocalBackend: async () => {
        await advanceBootProgress("backend.runtime", "Resolving Hermes runtime", 28);
        return resolveHermesBackend(backendArgs);
      },
      resolveRemote: () => {
        attemptedRemote = primaryBackendIsRemote();
        return resolveRemoteBackend(primaryProfileKey());
      },
      waitForDecision: waitForFirstRunSetupChoice,
      // Mutual exclusion with an in-app update (#50238). Remote connections
      // return before this waiter; local starts park until the updater exits.
      waitForLocalStart: waitForUpdateToFinish
    });
    if (setup.kind === "remote") {
      return setup.connection;
    }
    const backend = setup.backend;
    backend.args = getBackendArgsForRuntime(backend);
    const hermesCwd = resolveHermesCwd();
    const webDist = resolveWebDist();
    const readyFile = backend.readyFile ? makeDashboardReadyFile() : null;
    await advanceBootProgress("backend.spawn", `Starting Hermes backend via ${backend.label}`, 84);
    rememberLog(`Starting Hermes backend via ${backend.label}`);
    const hermesProcess = spawn5(
      backend.command,
      backend.args,
      hiddenWindowsChildOptions({
        cwd: hermesCwd,
        env: {
          ...process.env,
          // Explicitly pin HERMES_HOME for the child so Python's get_hermes_home()
          // resolves to the SAME location our resolveHermesHome() picked. Without
          // this pin, Python falls back to ~/.hermes on every platform — fine on
          // mac/linux (where our default matches), but on Windows our default is
          // %LOCALAPPDATA%\hermes, which differs from C:\Users\<u>\.hermes.
          // Mismatch would split config / sessions / .env / logs across two
          // directories. install.ps1 sets HERMES_HOME via setx; the desktop
          // can't reliably do that, so we set it inline for every spawn.
          HERMES_HOME,
          ...backend.env,
          TERMINAL_CWD: hermesCwd,
          HERMES_DASHBOARD_SESSION_TOKEN: token,
          // Marks this dashboard backend as desktop-spawned so it runs the cron
          // scheduler tick loop (the gateway isn't running under the app).
          HERMES_DESKTOP: "1",
          HERMES_WEB_DIST: webDist,
          ...readyFile ? { HERMES_DESKTOP_READY_FILE: readyFile } : {}
        },
        shell: backend.shell,
        stdio: ["ignore", "pipe", "pipe"]
      })
    );
    const processOwner = backendConnectionState.attachProcess(connectionAttempt, hermesProcess);
    if (!processOwner) {
      stopBackendChild2(hermesProcess);
      throw new Error("Hermes backend start was superseded by a newer connection attempt.");
    }
    hermesProcess.stdout.on("data", rememberLog);
    hermesProcess.stderr.on("data", rememberLog);
    let backendReady = false;
    let rejectBackendStart = null;
    const backendStartFailed = new Promise((_resolve, reject) => {
      rejectBackendStart = reject;
    });
    hermesProcess.once("error", (error) => {
      if (!backendConnectionState.clearForCurrentProcess(processOwner)) {
        rememberLog(`Ignoring stale Hermes backend error: ${error.message}`);
        rejectBackendStart?.(new Error("Hermes backend start was superseded by a newer connection attempt."));
        return;
      }
      rememberLog(`Hermes backend failed to start: ${error.message}`);
      updateBootProgress(
        {
          error: error.message,
          message: `Hermes backend failed to start: ${error.message}`,
          phase: "backend.error",
          running: false
        },
        { allowDecrease: true }
      );
      sendBackendExit({ code: null, signal: null, error: error.message });
      rejectBackendStart?.(error);
    });
    hermesProcess.once("exit", (code, signal) => {
      if (!backendConnectionState.clearForCurrentProcess(processOwner)) {
        rememberLog(`Ignoring stale Hermes backend exit (${signal || code})`);
        if (!backendReady) {
          rejectBackendStart?.(new Error("Hermes backend start was superseded by a newer connection attempt."));
        }
        return;
      }
      rememberLog(`Hermes backend exited (${signal || code})`);
      sendBackendExit({ code, signal });
      if (!backendReady) {
        const message = `Hermes backend exited before it became ready (${signal || code}).`;
        updateBootProgress(
          {
            error: message,
            message,
            phase: "backend.error",
            running: false
          },
          { allowDecrease: true }
        );
        rejectBackendStart?.(
          new Error(
            `Hermes backend exited before it became ready (${signal || code}). Log: ${DESKTOP_LOG_PATH}
${recentHermesLog()}`
          )
        );
      }
    });
    await advanceBootProgress("backend.port", "Waiting for Hermes backend to launch", 86);
    const port = await Promise.race([
      waitForDashboardPortAnnouncement(hermesProcess, { readyFile }),
      backendStartFailed
    ]);
    if (readyFile) {
      fs19.unlink(readyFile, () => {
      });
    }
    const baseUrl = `http://127.0.0.1:${port}`;
    await advanceBootProgress("backend.wait", "Waiting for Hermes backend to become ready", 90);
    await Promise.race([waitForHermes(baseUrl, token), backendStartFailed]);
    backendReady = true;
    backendStartFailure = null;
    const authToken = await adoptServedDashboardToken(baseUrl, token, {
      childAlive: () => hermesProcess.exitCode === null && !hermesProcess.killed,
      rememberLog
    });
    const wsUrl = `ws://127.0.0.1:${port}/api/ws?token=${encodeURIComponent(authToken)}`;
    const wsProbe = await probeGatewayWebSocket(wsUrl, { WebSocketImpl: globalThis.WebSocket });
    if (!wsProbe.ok) {
      throw new Error(
        `Local Hermes backend is HTTP-reachable but the WebSocket (/api/ws) rejected the session token: ${wsProbe.reason}`
      );
    }
    updateBootProgress({
      phase: "backend.ready",
      message: "Hermes backend is ready. Finalizing desktop startup",
      progress: 94,
      running: true,
      error: null
    });
    bootstrapRepairAttempt = 0;
    return {
      baseUrl,
      mode: "local",
      source: "local",
      authMode: "token",
      token: authToken,
      wsUrl,
      logs: hermesLog.slice(-80),
      ...getWindowState()
    };
  })().catch((error) => {
    if (!backendConnectionState.clearPromiseForAttempt(connectionAttempt)) {
      throw error;
    }
    if (error instanceof FirstRunSetupResetError) {
      throw error;
    }
    const message = error instanceof Error ? error.message : String(error);
    if (shouldLatchBackendStartFailure({ attemptedRemote })) {
      backendStartFailure = error instanceof Error ? error : new Error(message);
    }
    if (shouldLatchRemoteReauthFailure({ attemptedRemote, isReauth: isReauthRequiredError(error) })) {
      remoteReauthFailure = error instanceof Error ? error : new Error(message);
    }
    updateBootProgress(
      {
        error: message,
        message: `Desktop boot failed: ${message}`,
        phase: "backend.error",
        running: false
      },
      { allowDecrease: true }
    );
    throw error;
  });
  backendConnectionState.setPromise(connectionAttempt, connectionPromise);
  return connectionPromise;
}
function wireCommonWindowHandlers(win, { zoom = true } = {}) {
  installPreviewShortcut(win);
  installDevToolsShortcut(win);
  if (zoom) {
    installZoomShortcuts(win);
    installZoomReassertOnWindowEvents(win, () => restorePersistedZoomLevel(win));
    win.webContents.on("did-finish-load", () => restorePersistedZoomLevel(win));
  }
  installContextMenu(win);
  win.webContents.setWindowOpenHandler((details) => {
    openExternalUrl(details.url);
    return { action: "deny" };
  });
  win.webContents.on("will-navigate", (event, url) => {
    if (DEV_SERVER && url.startsWith(DEV_SERVER) || !DEV_SERVER && url.startsWith("file:")) {
      return;
    }
    event.preventDefault();
    openExternalUrl(url);
  });
}
var sessionWindows = createSessionWindowRegistry();
function focusWindow(win) {
  if (!win || win.isDestroyed()) {
    return;
  }
  if (win.isMinimized()) {
    win.restore();
  }
  if (!win.isVisible()) {
    win.show();
  }
  win.focus();
}
function spawnSecondaryWindow({ sessionId, watch } = {}) {
  const icon = getAppIconPath();
  const win = new BrowserWindow2({
    width: SESSION_WINDOW_MIN_WIDTH,
    height: SESSION_WINDOW_MIN_HEIGHT,
    minWidth: SESSION_WINDOW_MIN_WIDTH,
    minHeight: SESSION_WINDOW_MIN_HEIGHT,
    title: "Hermes",
    titleBarStyle: "hidden",
    titleBarOverlay: getTitleBarOverlayOptions(),
    trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : void 0,
    vibrancy: IS_MAC ? "sidebar" : void 0,
    opacity: windowOpacity(),
    icon,
    // Don't show until the renderer's first themed paint is ready. macOS
    // `vibrancy` ignores `backgroundColor` and paints a translucent OS
    // material (which follows the OS appearance, not the app theme), so a
    // dark-themed app on a light-mode Mac flashes white until the renderer
    // covers it. ready-to-show fires after the boot-time paint in
    // themes/context.tsx, so the window appears already themed.
    show: false,
    backgroundColor: getWindowBackgroundColor(),
    webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
  });
  if (IS_MAC) {
    win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION);
  }
  win.once("ready-to-show", () => {
    if (!win.isDestroyed()) {
      win.show();
    }
  });
  win.on("enter-full-screen", () => sendWindowStateChanged(true));
  win.on("leave-full-screen", () => sendWindowStateChanged(false));
  streamThrottle.register(win);
  wireCommonWindowHandlers(win, zoomWiringForWindowKind("chat"));
  loadWindowUrl(
    win,
    buildSessionWindowUrl(sessionId, {
      devServer: DEV_SERVER,
      rendererIndexPath: DEV_SERVER ? void 0 : resolveRendererIndex(),
      watch
    }),
    "Session window"
  );
  return win;
}
function createSessionWindow(sessionId, { watch = false } = {}) {
  return sessionWindows.openOrFocus(sessionId, () => spawnSecondaryWindow({ sessionId, watch }));
}
var instanceWindows = /* @__PURE__ */ new Set();
function nextInstanceBounds() {
  const source = BrowserWindow2.getFocusedWindow() || mainWindow;
  const fallback = computeWindowOptions(readWindowState(), screen2.getAllDisplays());
  const base = source && !source.isDestroyed() ? source.getBounds() : null;
  return instanceWindowBounds(base, fallback);
}
function createInstanceWindow() {
  const icon = getAppIconPath();
  const win = new BrowserWindow2({
    ...nextInstanceBounds(),
    minWidth: MIN_WIDTH,
    minHeight: MIN_HEIGHT,
    title: "Hermes",
    titleBarStyle: "hidden",
    titleBarOverlay: getTitleBarOverlayOptions(),
    trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : void 0,
    vibrancy: IS_MAC ? "sidebar" : void 0,
    opacity: windowOpacity(),
    icon,
    show: false,
    backgroundColor: getWindowBackgroundColor(),
    webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
  });
  instanceWindows.add(win);
  if (IS_MAC) {
    win.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION);
  }
  win.once("ready-to-show", () => {
    if (!win.isDestroyed()) {
      win.show();
    }
  });
  win.on("enter-full-screen", () => sendWindowStateChanged(true, win));
  win.on("leave-full-screen", () => sendWindowStateChanged(false, win));
  streamThrottle.register(win);
  wireCommonWindowHandlers(win, zoomWiringForWindowKind("chat"));
  win.on("closed", () => {
    instanceWindows.delete(win);
  });
  loadWindowUrl(win, DEV_SERVER || pathToFileURL3(resolveRendererIndex()).toString(), "Instance window");
  return win;
}
var wakeIndicatorController = createWakeIndicatorWindowController({
  devServer: DEV_SERVER,
  isMac: IS_MAC,
  loadWindowUrl,
  preloadPath: PRELOAD_PATH,
  rendererIndex: resolveRendererIndex,
  wireWindow: (window2) => wireCommonWindowHandlers(window2, zoomWiringForWindowKind("wakeIndicator"))
});
var petOverlayWindow = null;
function petOverlayUrl() {
  if (DEV_SERVER) {
    return `${DEV_SERVER.endsWith("/") ? DEV_SERVER.slice(0, -1) : DEV_SERVER}/?win=overlay#/`;
  }
  return `${pathToFileURL3(resolveRendererIndex()).toString()}?win=overlay#/`;
}
function spawnPetOverlayWindow(bounds) {
  const win = new BrowserWindow2({
    width: Math.max(80, Math.round(bounds?.width || 220)),
    height: Math.max(80, Math.round(bounds?.height || 220)),
    x: Number.isFinite(bounds?.x) ? Math.round(bounds.x) : void 0,
    y: Number.isFinite(bounds?.y) ? Math.round(bounds.y) : void 0,
    frame: false,
    transparent: true,
    resizable: false,
    movable: true,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    // Windows/Linux need this so the helper window does not get its own
    // taskbar/alt-tab entry. On macOS, cmd-tab is app-level and this can make
    // the whole app look like it vanished when the only newly-created visible
    // window is a frameless overlay. Use NSPanel + Mission Control hiding below
    // instead, leaving the main Hermes app as the Dock/cmd-tab anchor.
    skipTaskbar: !IS_MAC,
    hasShadow: false,
    alwaysOnTop: true,
    // macOS panels are non-activating helper windows and can float over full
    // screen spaces without becoming the app's main switcher window.
    type: IS_MAC ? "panel" : void 0,
    hiddenInMissionControl: IS_MAC,
    // Non-activating: the overlay must never become the app's key/main window,
    // or it (a frameless, taskbar-skipping panel) becomes the app's switcher
    // anchor and the Hermes icon drops out of cmd/alt-tab — especially when the
    // main window is minimized. We flip this on only while the composer needs
    // the keyboard (see hermes:pet-overlay:set-focusable).
    focusable: false,
    show: false,
    // Fully transparent — the renderer paints only the sprite + bubble.
    backgroundColor: "#00000000",
    webPreferences: {
      preload: PRELOAD_PATH,
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      devTools: true,
      // Keep the sprite animating + bubble updating while the main window is
      // minimized/blurred — the whole point of the overlay.
      backgroundThrottling: false
    }
  });
  win.setAlwaysOnTop(true, IS_MAC ? "floating" : "screen-saver");
  win.setHiddenInMissionControl?.(true);
  try {
    win.setVisibleOnAllWorkspaces(
      true,
      IS_MAC ? { visibleOnFullScreen: true, skipTransformProcessType: true } : void 0
    );
  } catch {
  }
  wireCommonWindowHandlers(win, zoomWiringForWindowKind("petOverlay"));
  win.once("ready-to-show", () => {
    if (!win.isDestroyed()) {
      win.showInactive();
    }
  });
  win.on("closed", () => {
    if (petOverlayWindow === win) {
      petOverlayWindow = null;
    }
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send("hermes:pet-overlay:control", { type: "pop-in" });
    }
  });
  loadWindowUrl(win, petOverlayUrl(), "Pet overlay");
  return win;
}
function openPetOverlay(bounds) {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    if (bounds) {
      petOverlayWindow.setBounds({
        x: Math.round(bounds.x),
        y: Math.round(bounds.y),
        width: Math.max(80, Math.round(bounds.width)),
        height: Math.max(80, Math.round(bounds.height))
      });
    }
    petOverlayWindow.showInactive();
    return petOverlayWindow;
  }
  petOverlayWindow = spawnPetOverlayWindow(bounds);
  return petOverlayWindow;
}
function closePetOverlay() {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    petOverlayWindow.close();
  }
  petOverlayWindow = null;
}
var QUICK_ENTRY_CONFIG_PATH = path22.join(app2.getPath("userData"), "quick-entry.json");
var quickEntryWindow = null;
var quickEntryLastState = null;
function readQuickEntrySettings() {
  try {
    return sanitizeQuickEntrySettings(JSON.parse(fs19.readFileSync(QUICK_ENTRY_CONFIG_PATH, "utf8")));
  } catch {
    return sanitizeQuickEntrySettings(void 0);
  }
}
function writeQuickEntrySettings(settings) {
  try {
    fs19.mkdirSync(path22.dirname(QUICK_ENTRY_CONFIG_PATH), { recursive: true });
    fs19.writeFileSync(QUICK_ENTRY_CONFIG_PATH, JSON.stringify(settings, null, 2), "utf8");
  } catch (error) {
    rememberLog(`[quick-entry] write failed: ${error.message}`);
  }
}
function quickEntryUrl() {
  if (DEV_SERVER) {
    return `${DEV_SERVER.endsWith("/") ? DEV_SERVER.slice(0, -1) : DEV_SERVER}/?win=quick#/`;
  }
  return `${pathToFileURL3(resolveRendererIndex()).toString()}?win=quick#/`;
}
function spawnQuickEntryWindow() {
  const cursor = screen2.getCursorScreenPoint();
  const display = screen2.getDisplayNearestPoint(cursor);
  const bounds = quickEntryWindowBounds(display?.workArea);
  const win = new BrowserWindow2({
    ...bounds,
    frame: false,
    transparent: true,
    resizable: false,
    movable: true,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    // Same rationale as the pet overlay: on Windows/Linux keep the helper out
    // of the taskbar/alt-tab list; on macOS use an NSPanel so the frameless
    // capture window never becomes the app's cmd-tab anchor.
    skipTaskbar: !IS_MAC,
    hasShadow: true,
    alwaysOnTop: true,
    type: IS_MAC ? "panel" : void 0,
    hiddenInMissionControl: IS_MAC,
    show: false,
    backgroundColor: "#00000000",
    webPreferences: {
      preload: PRELOAD_PATH,
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      devTools: true
    }
  });
  win.setAlwaysOnTop(true, IS_MAC ? "floating" : "screen-saver");
  win.setHiddenInMissionControl?.(true);
  try {
    win.setVisibleOnAllWorkspaces(
      true,
      IS_MAC ? { visibleOnFullScreen: true, skipTransformProcessType: true } : void 0
    );
  } catch {
  }
  wireCommonWindowHandlers(win, zoomWiringForWindowKind("quickEntry"));
  win.on("blur", () => {
    if (!win.isDestroyed()) {
      win.hide();
    }
  });
  win.on("closed", () => {
    if (quickEntryWindow === win) {
      quickEntryWindow = null;
    }
  });
  win.webContents.on("did-finish-load", () => {
    if (!win.isDestroyed() && quickEntryLastState) {
      win.webContents.send("hermes:quick-entry:state", quickEntryLastState);
    }
  });
  loadWindowUrl(win, quickEntryUrl(), "Quick entry");
  return win;
}
function repositionQuickEntryWindow(win) {
  try {
    const display = screen2.getDisplayNearestPoint(screen2.getCursorScreenPoint());
    win.setBounds(quickEntryWindowBounds(display?.workArea));
  } catch (error) {
    rememberLog(`[quick-entry] reposition failed: ${error.message}`);
  }
}
function showQuickEntryWindow() {
  if (!quickEntryWindow || quickEntryWindow.isDestroyed()) {
    quickEntryWindow = spawnQuickEntryWindow();
    quickEntryWindow.once("ready-to-show", () => {
      if (!quickEntryWindow?.isDestroyed()) {
        quickEntryWindow.show();
        quickEntryWindow.focus();
      }
    });
    return;
  }
  repositionQuickEntryWindow(quickEntryWindow);
  quickEntryWindow.show();
  quickEntryWindow.focus();
  quickEntryWindow.webContents.send("hermes:quick-entry:shown");
}
function hideQuickEntryWindow() {
  if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
    quickEntryWindow.hide();
  }
}
function toggleQuickEntryWindow() {
  if (quickEntryWindow && !quickEntryWindow.isDestroyed() && quickEntryWindow.isVisible()) {
    hideQuickEntryWindow();
    return;
  }
  showQuickEntryWindow();
}
var quickEntryShortcut = createQuickEntryShortcut(globalShortcut, toggleQuickEntryWindow);
function applyQuickEntrySettings(settings) {
  const state = quickEntryShortcut.apply(settings);
  if (!settings.enabled) {
    if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
      quickEntryWindow.close();
    }
    quickEntryWindow = null;
  }
  if (state.error === "taken") {
    rememberLog(`[quick-entry] shortcut ${state.shortcut} is already taken by another application`);
  } else if (state.error === "invalid") {
    rememberLog(`[quick-entry] shortcut ${state.shortcut} is not a valid accelerator`);
  }
  return { ...state, enabled: settings.enabled };
}
function closeQuickEntryWindow() {
  quickEntryShortcut.dispose();
  if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
    quickEntryWindow.close();
  }
  quickEntryWindow = null;
}
function createWindow() {
  const icon = getAppIconPath();
  const savedWindowState = readWindowState();
  mainWindow = new BrowserWindow2({
    ...computeWindowOptions(savedWindowState, screen2.getAllDisplays()),
    minWidth: MIN_WIDTH,
    minHeight: MIN_HEIGHT,
    title: "Hermes",
    // Frameless title bar on every platform so the renderer can paint the
    // "hide sidebar" button (and other left-side titlebar tools) flush with
    // the top edge — matching the macOS layout where the traffic lights sit
    // inside the same band. On Windows/Linux, titleBarOverlay tells Electron
    // to paint native min/max/close in the top-right of the renderer; on
    // macOS it just reserves a content inset alongside the traffic lights.
    titleBarStyle: "hidden",
    titleBarOverlay: getTitleBarOverlayOptions(),
    trafficLightPosition: IS_MAC ? WINDOW_BUTTON_POSITION : void 0,
    vibrancy: IS_MAC ? "sidebar" : void 0,
    opacity: windowOpacity(),
    icon,
    // Hidden until the first themed paint so macOS `vibrancy` (which ignores
    // `backgroundColor` and follows the OS appearance) can't flash a light
    // material before the renderer paints the app theme. See createSessionWindow.
    show: false,
    backgroundColor: getWindowBackgroundColor(),
    // Shared with the secondary session windows (chatWindowWebPreferences);
    // stream-aware throttling is applied per-window via streamThrottle so a
    // live answer keeps painting while the window is blurred or minimized,
    // without pinning visibilityState to 'visible' at idle. See
    // session-windows.ts and stream-throttle.ts.
    webPreferences: chatWindowWebPreferences(PRELOAD_PATH)
  });
  if (IS_MAC) {
    mainWindow.setWindowButtonPosition?.(WINDOW_BUTTON_POSITION);
    if (icon) {
      app2.dock?.setIcon(icon);
    }
  }
  if (!IS_MAC) {
    if (!nativeThemeListenerInstalled) {
      nativeThemeListenerInstalled = true;
      nativeTheme.on("updated", () => {
        for (const win of BrowserWindow2.getAllWindows()) {
          applyTitleBarOverlay(win);
        }
      });
    }
  }
  if (savedWindowState?.isMaximized) {
    mainWindow.maximize();
  }
  mainWindow.once("ready-to-show", () => {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.show();
    }
    schedulePersistWindowState();
    if (IS_WINDOWS3) {
      try {
        writeSandboxMarker(
          app2.getPath("userData"),
          markerAfterSuccessfulBoot({
            fallbackActive: windowsSandboxFallbackSticky,
            reason: windowsSandboxFallbackReason,
            appVersion: app2.getVersion()
          })
        );
      } catch (error) {
        rememberLog(`[sandbox] marker update after ready-to-show failed: ${error?.message || error}`);
      }
    }
  });
  if (process.env.TEST_WORKER_INDEX !== void 0) {
    if (mainWindow && !mainWindow.isDestroyed() && !mainWindow.isVisible()) {
      mainWindow.show();
    }
  }
  mainWindow.on("will-enter-full-screen", () => sendWindowStateChanged(true));
  mainWindow.on("enter-full-screen", () => sendWindowStateChanged(true));
  mainWindow.on("will-leave-full-screen", () => sendWindowStateChanged(false));
  mainWindow.on("leave-full-screen", () => sendWindowStateChanged(false));
  mainWindow.on("minimize", () => sendWindowStateChanged());
  mainWindow.on("restore", () => sendWindowStateChanged());
  mainWindow.on("hide", () => sendWindowStateChanged());
  mainWindow.on("show", () => sendWindowStateChanged());
  mainWindow.on("resized", schedulePersistWindowState);
  mainWindow.on("moved", schedulePersistWindowState);
  mainWindow.on("maximize", schedulePersistWindowState);
  mainWindow.on("unmaximize", schedulePersistWindowState);
  mainWindow.on("close", () => schedulePersistWindowState.flush());
  mainWindow.on("close", (event) => {
    if (isTrayEnabled() && !isReallyQuitting) {
      event.preventDefault();
      mainWindow?.hide();
    }
  });
  const createdMainWindow = mainWindow;
  mainWindow.on("closed", () => {
    closePetOverlay();
    wakeIndicatorController.close();
    if (mainWindow === createdMainWindow) {
      mainWindow = null;
      _rendererReadyForDeepLink = false;
    }
  });
  streamThrottle.register(mainWindow);
  wireCommonWindowHandlers(mainWindow, zoomWiringForWindowKind("chat"));
  mainWindow.webContents.on("render-process-gone", (_event, details) => {
    rememberLog(`[renderer] render-process-gone reason=${details?.reason} exitCode=${details?.exitCode}`);
    if (details?.reason === "crashed" || details?.reason === "oom") {
      const now = Date.now();
      rendererReloadTimes = rendererReloadTimes.filter((t2) => now - t2 < RENDERER_RELOAD_WINDOW_MS);
      if (rendererReloadTimes.length >= RENDERER_RELOAD_MAX) {
        rememberLog(
          `[renderer] suppressing reload: ${rendererReloadTimes.length} crashes within ${RENDERER_RELOAD_WINDOW_MS}ms (likely a crash loop)`
        );
        if (shouldRelaunchForRendererSandboxCrashLoop({
          reason: details?.reason,
          exitCode: details?.exitCode,
          alreadyNoSandbox: windowsSandboxFallbackActive || alreadyHasNoSandbox(process.argv, process.env),
          relaunchAttempted: windowsNoSandboxRelaunchAttempted
        })) {
          windowsNoSandboxRelaunchAttempted = true;
          windowsSandboxFallbackActive = true;
          windowsSandboxFallbackSticky = true;
          windowsSandboxFallbackReason = "renderer-crash-loop";
          try {
            writeSandboxMarker(app2.getPath("userData"), fallbackMarker("renderer-crash-loop", app2.getVersion()));
          } catch {
          }
          rememberLog("[renderer] Windows sandbox crash loop detected; relaunching once with --no-sandbox (#38216)");
          try {
            app2.relaunch({ args: buildNoSandboxRelaunchArgs(process.argv.slice(1)) });
            app2.exit(0);
          } catch (err) {
            rememberLog(`[renderer] --no-sandbox relaunch failed: ${err?.message || err}`);
          }
        }
        return;
      }
      rendererReloadTimes.push(now);
      setImmediate(() => {
        if (!mainWindow || mainWindow.isDestroyed()) {
          return;
        }
        try {
          mainWindow.webContents.reload();
        } catch (err) {
          rememberLog(`[renderer] reload after crash failed: ${err?.message || err}`);
        }
      });
    }
  });
  mainWindow.webContents.on("unresponsive", () => rememberLog("[renderer] webContents became unresponsive"));
  mainWindow.webContents.on("console-message", (_event, detailsOrLevel, message, line, sourceId) => {
    const details = detailsOrLevel && typeof detailsOrLevel === "object" ? detailsOrLevel : null;
    const level = details ? details.level : detailsOrLevel;
    if (level !== 3) {
      return;
    }
    const text = details ? details.message : message;
    const src = details ? details.sourceUrl : sourceId;
    const lineNo = details ? details.lineNumber : line;
    rememberLog(`[renderer console] ${text} (${src}:${lineNo})`);
  });
  loadWindowUrl(mainWindow, DEV_SERVER || pathToFileURL3(resolveRendererIndex()).toString(), "Renderer");
  startHermes().catch((error) => rememberLog(error.stack || error.message));
  mainWindow.webContents.once("did-finish-load", () => {
    broadcastBootProgress();
    sendWindowStateChanged();
  });
}
ipcMain.handle("hermes:connection", async (_event, profile) => ensureBackend(profile));
ipcMain.handle("hermes:connection:revalidate", async () => {
  const connectionPromise = backendConnectionState.getPromise();
  if (!connectionPromise) {
    await revalidatePool();
    return { ok: true, rebuilt: false };
  }
  return remoteRevalidation.run(connectionPromise, async () => {
    const [result] = await Promise.all([
      revalidateRemoteConnection({
        connectionPromise,
        currentConnectionPromise: () => backendConnectionState.getPromise(),
        log: rememberLog,
        probe: fetchPublicJson,
        resetConnection: resetHermesConnection,
        tracker: remoteLiveness
      }),
      revalidatePool()
    ]);
    if (result.rebuilt) {
      const conn = await connectionPromise.catch(() => null);
      if (conn?.remoteKind === "ssh") {
        const profile = primaryProfileKey();
        await sshBootstrapCoordinator.cancelAndWait(sshScopeKey(profile));
        await teardownSshConnection(profile);
      }
    }
    return result;
  });
});
function revalidatePool() {
  return revalidatePooledRemoteBackends({
    entries: backendPool.entries(),
    log: rememberLog,
    probe: fetchPublicJson,
    stopBackend: stopPoolBackend,
    tracker: remoteLiveness
  });
}
ipcMain.handle("hermes:backend:touch", async (_event, profile) => {
  touchPoolBackend(profile);
  return { ok: true };
});
ipcMain.handle("hermes:gateway:ws-url", async (_event, profile) => {
  return gatewayWsUrlIpcResult(() => freshGatewayWsUrl(profile));
});
ipcMain.handle("hermes:window:openSession", async (_event, sessionId, opts) => {
  if (typeof sessionId !== "string" || !sessionId.trim()) {
    return { ok: false, error: "invalid-session-id" };
  }
  createSessionWindow(sessionId.trim(), { watch: opts?.watch === true });
  return { ok: true };
});
ipcMain.handle("hermes:window:openInstance", async () => {
  createInstanceWindow();
  return { ok: true };
});
ipcMain.handle("hermes:wake-indicator:get", () => wakeIndicatorController.getState());
ipcMain.on("hermes:wake-indicator:set", (_event, state) => {
  wakeIndicatorController.setState(state);
});
ipcMain.handle("hermes:zoom:get", (event) => {
  const window2 = BrowserWindow2.fromWebContents(event.sender);
  const level = window2 && !window2.isDestroyed() ? window2.webContents.getZoomLevel() : DEFAULT_ZOOM_LEVEL;
  return { level, percent: zoomLevelToPercent(level) };
});
ipcMain.on("hermes:zoom:set-percent", (event, percent) => {
  const window2 = BrowserWindow2.fromWebContents(event.sender);
  if (!window2 || window2.isDestroyed()) {
    return;
  }
  setAndPersistZoomLevel(window2, percentToZoomLevel(Number(percent)));
});
ipcMain.handle("hermes:pet-overlay:open", async (_event, request2) => {
  const bounds = request2 && request2.bounds ? request2.bounds : request2;
  const isScreen = Boolean(request2 && request2.screen);
  let screenBounds = bounds;
  try {
    if (bounds && !isScreen && mainWindow && !mainWindow.isDestroyed()) {
      const content = mainWindow.getContentBounds();
      screenBounds = {
        x: content.x + (bounds.x || 0),
        y: content.y + (bounds.y || 0),
        width: bounds.width,
        height: bounds.height
      };
    }
  } catch {
  }
  openPetOverlay(screenBounds);
  return { ok: true, bounds: screenBounds };
});
ipcMain.handle("hermes:pet-overlay:close", async () => {
  closePetOverlay();
  return { ok: true };
});
ipcMain.on("hermes:pet-overlay:set-bounds", (_event, bounds) => {
  if (!petOverlayWindow || petOverlayWindow.isDestroyed() || !bounds) {
    return;
  }
  const win = petOverlayWindow;
  const width = Math.max(80, Math.round(bounds.width));
  const height = Math.max(80, Math.round(bounds.height));
  const [curW, curH] = win.getSize();
  const resizing = width !== curW || height !== curH;
  if (resizing && !win.isResizable()) {
    win.setResizable(true);
  }
  win.setBounds({ x: Math.round(bounds.x), y: Math.round(bounds.y), width, height });
  if (resizing) {
    win.setResizable(false);
  }
});
ipcMain.on("hermes:pet-overlay:ignore-mouse", (_event, ignore) => {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    petOverlayWindow.setIgnoreMouseEvents(Boolean(ignore), { forward: true });
  }
});
ipcMain.on("hermes:pet-overlay:set-focusable", (_event, focusable) => {
  if (!petOverlayWindow || petOverlayWindow.isDestroyed()) {
    return;
  }
  petOverlayWindow.setFocusable(Boolean(focusable));
  if (focusable) {
    petOverlayWindow.focus();
  }
});
ipcMain.on("hermes:pet-overlay:state", (_event, payload) => {
  if (petOverlayWindow && !petOverlayWindow.isDestroyed()) {
    petOverlayWindow.webContents.send("hermes:pet-overlay:state", payload);
  }
});
ipcMain.on("hermes:pet-overlay:control", (_event, payload) => {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  if (payload && payload.type === "toggle-app") {
    if (mainWindow.isMinimized() || !mainWindow.isVisible()) {
      mainWindow.show();
      mainWindow.focus();
    } else {
      mainWindow.minimize();
    }
    return;
  }
  if (payload && payload.type === "open-app") {
    if (mainWindow.isMinimized()) {
      mainWindow.restore();
    }
    mainWindow.show();
    mainWindow.focus();
  }
  mainWindow.webContents.send("hermes:pet-overlay:control", payload);
});
ipcMain.handle("hermes:bootstrap:reset", async () => {
  rememberLog("[bootstrap] reset requested by renderer; clearing latched failure");
  await teardownPrimaryBackendAndWait();
  bootstrapFailure = null;
  backendStartFailure = null;
  remoteReauthFailure = null;
  getFirstRunSetupGate().resetForRetry();
  resetBootstrapSnapshot();
  return { ok: true };
});
ipcMain.handle("hermes:bootstrap:repair", async () => {
  bootstrapRepairAttempt += 1;
  const primaryProc = backendConnectionState.getProcess();
  const primaryBackendAlive = Boolean(
    primaryProc && primaryProc.exitCode === null && primaryProc.signalCode === null
  );
  const repairDecision = decideBootstrapRepair({
    attempt: bootstrapRepairAttempt,
    maxSoftAttempts: MAX_BOOTSTRAP_REPAIR_SOFT_ATTEMPTS,
    primaryBackendAlive
  });
  rememberLog(
    `[bootstrap] repair requested by renderer; forcing reinstall + clearing latched failure (attempt=${repairDecision.attempt}/${MAX_BOOTSTRAP_REPAIR_SOFT_ATTEMPTS}, primaryBackendAlive=${primaryBackendAlive}, hardReinstall=${repairDecision.hardReinstall}): ${repairDecision.reason}`
  );
  bootstrapRepairRequested = repairDecision.hardReinstall;
  bootstrapFailure = null;
  backendStartFailure = null;
  remoteReauthFailure = null;
  getFirstRunSetupGate().resetForRepair();
  resetHermesConnection();
  return { ok: true };
});
ipcMain.handle("hermes:bootstrap:continue-local", async () => {
  rememberLog("[bootstrap] local install selected by renderer; continuing first-launch bootstrap");
  continueFirstRunLocalBootstrap();
  return { ok: true };
});
ipcMain.handle("hermes:bootstrap:cancel", async () => {
  if (bootstrapAbortController) {
    try {
      bootstrapAbortController.abort();
    } catch {
    }
    return { ok: true, cancelled: true };
  }
  return { ok: false, cancelled: false };
});
ipcMain.handle("hermes:boot-progress:get", async () => bootProgressState);
ipcMain.handle("hermes:bootstrap:get", async () => getBootstrapState());
ipcMain.handle(
  "hermes:connection-config:get",
  async (_event, profile) => sanitizeDesktopConnectionConfig(readDesktopConnectionConfig(), profile)
);
ipcMain.handle("hermes:ssh-config:hosts", async () => ({ hosts: collectSshConfigHosts() }));
ipcMain.handle("hermes:ssh-config:resolve", async (_event, host) => {
  const value = String(host || "").trim();
  if (!value) {
    throw new Error("SSH host is required.");
  }
  const ssh = process.platform === "win32" ? path22.join(process.env.SystemRoot || "C:\\Windows", "System32", "OpenSSH", "ssh.exe") : "ssh";
  return new Promise((resolve, reject) => {
    const child = spawn5(ssh, ["-G", "--", value], hiddenWindowsChildOptions({ stdio: ["ignore", "pipe", "pipe"] }));
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error("SSH config resolution timed out."));
    }, 1e4);
    child.stdout.on("data", (chunk) => {
      stdout += String(chunk);
    });
    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        reject(new Error(stderr.trim() || "Could not resolve SSH host."));
      } else {
        resolve(parseSshGOutput(stdout));
      }
    });
  });
});
ipcMain.handle("hermes:connection-config:test", async (_event, payload) => testDesktopConnectionConfig(payload));
ipcMain.handle("hermes:connection-config:probe", async (_event, rawUrl) => probeRemoteAuthMode(rawUrl));
ipcMain.handle("hermes:connection-config:oauth-login", async (_event, rawUrl) => {
  const baseUrl = normalizeRemoteBaseUrl(rawUrl);
  let statusBody = null;
  try {
    statusBody = await fetchPublicJson(`${baseUrl}/api/status`, { timeoutMs: 8e3 });
  } catch {
  }
  const strategy = resolveLoginStrategy(statusBody);
  if (strategy === "native") {
    try {
      const tokens = await runNativeLogin(baseUrl, {
        openExternal: (url) => shell.openExternal(url),
        postJson: (url, body, opts) => postJsonNoAuth(url, body, opts),
        rememberLog
      });
      _storeNativeTokens(baseUrl, tokens);
      remoteReauthFailure = null;
      return { ok: true, baseUrl, connected: true };
    } catch (error) {
      rememberLog(
        `[native-oauth] native login failed (${error instanceof Error ? error.message : String(error)}); falling back to embedded flow`
      );
    }
  }
  await openOauthLoginWindow(baseUrl);
  const connected = await hasOauthSessionCookie(baseUrl);
  if (connected) {
    remoteReauthFailure = null;
  }
  return { ok: true, baseUrl, connected };
});
ipcMain.handle("hermes:connection-config:oauth-logout", async (_event, rawUrl) => {
  const baseUrl = rawUrl ? normalizeRemoteBaseUrl(rawUrl) : "";
  await clearOauthSession(baseUrl || void 0);
  if (baseUrl) {
    _clearNativeTokens(baseUrl);
  }
  const connected = baseUrl ? await hasLiveOauthSession(baseUrl) || hasNativeSession(baseUrl) : false;
  return { ok: true, connected };
});
ipcMain.handle("hermes:cloud:status", async () => ({
  portalBaseUrl: resolvePortalBaseUrl(),
  signedIn: await hasLivePortalSession()
}));
ipcMain.handle("hermes:cloud:login", async () => {
  await openPortalLoginWindow();
  return { ok: true, signedIn: await hasLivePortalSession() };
});
ipcMain.handle("hermes:cloud:logout", async () => {
  await clearOauthSession(resolvePortalBaseUrl());
  return { ok: true, signedIn: await hasLivePortalSession() };
});
ipcMain.handle("hermes:cloud:discover", async (_event, org) => {
  return discoverCloudAgents(typeof org === "string" && org ? org : void 0);
});
ipcMain.handle("hermes:cloud:agent-sign-in", async (_event, dashboardUrl) => {
  return cloudAgentSilentSignIn(dashboardUrl);
});
ipcMain.handle("hermes:connection-config:save", async (_event, payload) => {
  const config = coerceDesktopConnectionConfig(payload);
  writeDesktopConnectionConfig(config);
  return sanitizeDesktopConnectionConfig(config, payload?.profile);
});
ipcMain.handle("hermes:connection-config:apply", async (_event, payload) => {
  const config = coerceDesktopConnectionConfig(payload);
  writeDesktopConnectionConfig(config);
  const key = connectionScopeKey(payload?.profile);
  const scope = key || "";
  await applyConnectionChange({
    cancelAndWait: (value) => sshBootstrapCoordinator.cancelAndWait(value),
    isPrimary: !key || key === primaryProfileKey(),
    rehomePrimary: () => rehomePrimaryConnection({
      clearLocalBootstrapFailure: () => {
        bootstrapFailure = null;
      },
      mode: config.mode,
      notifyConnectionApplied: sendConnectionApplied,
      resumeFirstRunRemote: abandonFirstRunSetupChoiceForRemoteApply,
      teardownPrimaryBackend: teardownPrimaryBackendAndWait
    }),
    scope,
    sendApplied: sendConnectionApplied,
    stopPool: stopPoolBackend,
    teardownPrimary: () => teardownPrimaryBackendAndWait({ soft: true }),
    teardownSsh: (value) => teardownSshConnection(value || null)
  });
  return sanitizeDesktopConnectionConfig(config, payload?.profile);
});
ipcMain.handle("hermes:profile:get", async () => ({ profile: readActiveDesktopProfile() }));
ipcMain.handle("hermes:profile:set", async (_event, name) => {
  const next = writeActiveDesktopProfile(name);
  await teardownPrimaryBackendAndWait();
  mainWindow?.reload();
  return { profile: next };
});
ipcMain.on("hermes:previewShortcutActive", (_event, active) => {
  previewShortcutActive = Boolean(active);
});
ipcMain.handle("hermes:requestMicrophoneAccess", async () => {
  if (!IS_MAC || typeof systemPreferences.askForMediaAccess !== "function") {
    return true;
  }
  return systemPreferences.askForMediaAccess("microphone");
});
async function interceptSessionRequestForRemote(request2) {
  if (typeof request2?.path !== "string") {
    return void 0;
  }
  const method = (request2.method || "GET").toUpperCase();
  let parsed;
  try {
    parsed = new URL(request2.path, "http://x");
  } catch {
    return void 0;
  }
  const { pathname, searchParams } = parsed;
  if (method === "GET" && pathname === "/api/profiles/sessions") {
    const remoteProfiles = configuredRemoteProfileNames();
    if (remoteProfiles.length === 0) {
      return void 0;
    }
    const requested = (searchParams.get("profile") || "all").trim() || "all";
    if (requested !== "all") {
      return profileHasRemoteOverride(requested) ? remoteSessionList(requested, searchParams) : void 0;
    }
    return mergeRemoteProfileSessions(searchParams, remoteProfiles);
  }
  if (method === "GET" && pathname === "/api/profiles/sessions/sidebar") {
    const remoteProfiles = configuredRemoteProfileNames();
    if (remoteProfiles.length === 0) {
      return void 0;
    }
    const recentsProfile = (searchParams.get("recents_profile") || "all").trim() || "all";
    const sliceParams = (limitKey, defaultLimit, extra) => {
      const sp = new URLSearchParams({
        limit: searchParams.get(limitKey) || defaultLimit,
        offset: "0",
        min_messages: "1",
        archived: "exclude",
        order: "recent",
        ...extra
      });
      return sp;
    };
    const recentsSp = sliceParams("recents_limit", "20", { profile: recentsProfile });
    const recentsExclude = searchParams.get("recents_exclude");
    if (recentsExclude) {
      recentsSp.set("exclude_sources", recentsExclude);
    }
    const cronSp = sliceParams("cron_limit", "50", { profile: "all", source: "cron" });
    const messagingSp = sliceParams("messaging_limit", "100", { profile: "all" });
    const messagingExclude = searchParams.get("messaging_exclude");
    if (messagingExclude) {
      messagingSp.set("exclude_sources", messagingExclude);
    }
    const [recents, cron, messaging] = await Promise.all([
      fetchProfilesSessionSlice(recentsSp, remoteProfiles),
      fetchProfilesSessionSlice(cronSp, remoteProfiles),
      fetchProfilesSessionSlice(messagingSp, remoteProfiles)
    ]);
    return {
      recents: {
        sessions: rowsOf(recents),
        total: Number(recents?.total) || 0,
        profile_totals: recents?.profile_totals || {}
      },
      cron: { sessions: rowsOf(cron) },
      messaging: {
        sessions: rowsOf(messaging),
        total: Number(messaging?.total) || rowsOf(messaging).length
      },
      errors: []
    };
  }
  if (/^\/api\/sessions\/[^/]+(\/messages)?$/.test(pathname)) {
    const profile = (searchParams.get("profile") || request2.profile || "").trim();
    if (!profile) {
      return void 0;
    }
    if (profileHasRemoteOverride(profile)) {
      if (method === "GET") {
        return fetchJsonForProfile(profile, pathname);
      }
      const body = request2.body && typeof request2.body === "object" ? { ...request2.body } : request2.body;
      if (body) {
        delete body.profile;
      }
      return requestJsonForProfile(profile, pathname, method, body);
    }
    if (globalRemoteActive()) {
      const sep = pathname.includes("?") ? "&" : "?";
      const path23 = `${pathname}${sep}profile=${encodeURIComponent(profile)}`;
      if (method === "GET") {
        return fetchJsonForProfile(null, path23);
      }
      const body = request2.body && typeof request2.body === "object" ? { ...request2.body, profile } : { profile };
      return requestJsonForProfile(null, path23, method, body);
    }
    return void 0;
  }
  return void 0;
}
var rowsOf = (data) => Array.isArray(data?.sessions) ? data.sessions : [];
async function remoteSessionList(profile, searchParams) {
  const qs = new URLSearchParams(searchParams);
  qs.delete("profile");
  const data = await fetchJsonForProfile(profile, `/api/sessions?${qs}`);
  for (const s of rowsOf(data)) {
    s.profile = profile;
    s.is_default_profile = false;
  }
  return { ...data, sessions: rowsOf(data) };
}
async function fetchProfilesSessionSlice(searchParams, remoteProfiles) {
  const requested = (searchParams.get("profile") || "all").trim() || "all";
  if (requested !== "all") {
    if (profileHasRemoteOverride(requested)) {
      return remoteSessionList(requested, searchParams);
    }
    return fetchPrimaryProfileSessions(searchParams, fetchJsonForProfile);
  }
  return mergeRemoteProfileSessions(searchParams, remoteProfiles);
}
async function mergeRemoteProfileSessions(searchParams, remoteProfiles) {
  const limit = Math.max(1, Number(searchParams.get("limit")) || 20);
  const offset = Math.max(0, Number(searchParams.get("offset")) || 0);
  const order = searchParams.get("order") === "created" ? "started_at" : "last_active";
  const base = await fetchPrimaryProfileSessions(searchParams, fetchJsonForProfile);
  const remoteParams = new URLSearchParams(searchParams);
  remoteParams.set("limit", String(limit + offset));
  remoteParams.set("offset", "0");
  const remoteSet = new Set(remoteProfiles);
  const merged = rowsOf(base).filter((s) => !remoteSet.has(s?.profile));
  const profileTotals = { ...base.profile_totals || {} };
  let total = (Number(base.total) || 0) - remoteProfiles.reduce((n, p2) => n + (profileTotals[p2] || 0), 0);
  await Promise.all(
    remoteProfiles.map(async (name) => {
      const list = await remoteSessionList(name, remoteParams).catch(() => null);
      if (!list) {
        delete profileTotals[name];
        return;
      }
      const rows = rowsOf(list);
      merged.push(...rows);
      profileTotals[name] = Number(list.total) || rows.length;
      total += profileTotals[name];
    })
  );
  const recency = (s) => s?.[order] ?? s?.started_at ?? 0;
  merged.sort((a, b2) => recency(b2) - recency(a));
  return { ...base, sessions: merged.slice(offset, offset + limit), total, profile_totals: profileTotals };
}
ipcMain.handle("hermes:api", async (_event, request2) => {
  const rerouted = await interceptSessionRequestForRemote(request2);
  if (rerouted !== void 0) {
    return rerouted;
  }
  const tornDownProfile = await prepareProfileDeleteRequest(request2);
  const profile = request2?.profile;
  const routeProfile = resolveRouteProfile(tornDownProfile, profile);
  const connection = await ensureBackend(routeProfile);
  const timeoutMs = resolveTimeoutMs(request2?.timeoutMs, DEFAULT_FETCH_TIMEOUT_MS);
  const requestPath = pathWithGlobalRemoteProfile(request2.path, profile, profileRouteOptions(profile));
  const url = `${connection.baseUrl}${requestPath}`;
  if (connection.authMode === "oauth") {
    if (request2?.upload) {
      throw new Error("File uploads are not supported against OAuth-gated remote backends yet.");
    }
    const nativeAt = await ensureNativeAccessToken(connection.baseUrl).catch(() => null);
    const restAuth = resolveOauthRestAuth(nativeAt);
    if (restAuth.kind === "bearer") {
      return fetchJson(url, null, {
        method: request2?.method,
        body: request2?.body,
        timeoutMs,
        bearer: restAuth.token
      });
    }
    return fetchJsonViaOauthSession(url, {
      method: request2?.method,
      body: request2?.body,
      timeoutMs
    });
  }
  return fetchJson(url, connection.token, {
    method: request2?.method,
    body: request2?.body,
    upload: request2?.upload,
    timeoutMs
  });
});
var isDuplicateNotification = createEventDeduper();
var claimedAmbientCue = createEventDeduper();
ipcMain.handle("hermes:ambient:claim", (_event, key) => !claimedAmbientCue(String(key ?? "")));
ipcMain.handle("hermes:tray:get", () => isTrayEnabled());
ipcMain.handle("hermes:tray:set", (_event, enabled2) => {
  return setTrayEnabled(Boolean(enabled2), {
    onShow: () => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.show();
        focusWindow(mainWindow);
      }
    },
    onQuit: () => {
      isReallyQuitting = true;
      app2.quit();
    }
  });
});
if (loadTrayPrefs()) {
  setTrayEnabled(true, {
    onShow: () => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.show();
        focusWindow(mainWindow);
      }
    },
    onQuit: () => {
      isReallyQuitting = true;
      app2.quit();
    }
  });
}
ipcMain.handle("hermes:notify", (_event, payload) => {
  if (!Notification.isSupported()) {
    return false;
  }
  if (isDuplicateNotification(`${payload?.kind ?? ""}:${payload?.sessionId ?? ""}`)) {
    return true;
  }
  const actions = Array.isArray(payload?.actions) ? payload.actions : [];
  const notification = new Notification({
    title: payload?.title || "Hermes",
    body: payload?.body || "",
    silent: Boolean(payload?.silent),
    actions: actions.map((action) => ({ type: "button", text: String(action?.text || "") }))
  });
  notification.on("click", () => {
    if (!mainWindow || mainWindow.isDestroyed()) {
      return;
    }
    focusWindow(mainWindow);
    if (payload?.sessionId) {
      mainWindow.webContents.send("hermes:focus-session", payload.sessionId);
    }
  });
  notification.on("action", (_actionEvent, index) => {
    if (!mainWindow || mainWindow.isDestroyed()) {
      return;
    }
    const action = actions[index];
    if (action?.id) {
      mainWindow.webContents.send("hermes:notification-action", { sessionId: payload?.sessionId, actionId: action.id });
    }
  });
  notification.show();
  return true;
});
var DATA_URL_READ_MAX_CONFIG_PATH = path22.join(app2.getPath("userData"), "data-url-read-max.json");
function readPersistedDataUrlReadMaxMb() {
  try {
    return clampDataUrlReadMaxMb(JSON.parse(fs19.readFileSync(DATA_URL_READ_MAX_CONFIG_PATH, "utf8")).maxMb);
  } catch {
    return DATA_URL_READ_DEFAULT_MAX_MB;
  }
}
var dataUrlReadMaxMb = readPersistedDataUrlReadMaxMb();
function persistDataUrlReadMaxMb(maxMb) {
  const next = clampDataUrlReadMaxMb(maxMb);
  dataUrlReadMaxMb = next;
  try {
    fs19.mkdirSync(path22.dirname(DATA_URL_READ_MAX_CONFIG_PATH), { recursive: true });
    fs19.writeFileSync(DATA_URL_READ_MAX_CONFIG_PATH, JSON.stringify({ maxMb: next }, null, 2), "utf8");
  } catch (error) {
    rememberLog(`[data-url-read-max] write failed: ${error.message}`);
  }
  return next;
}
ipcMain.handle("hermes:data-url-read-max:get", () => ({
  maxMb: dataUrlReadMaxMb,
  // Keep the default bytes constant visible for tests / diagnostics.
  defaultMaxMb: DATA_URL_READ_DEFAULT_MAX_MB,
  maxBytes: dataUrlReadMaxBytesFromMb(dataUrlReadMaxMb)
}));
ipcMain.handle("hermes:data-url-read-max:set", (_event, maxMb) => {
  const next = persistDataUrlReadMaxMb(maxMb);
  return {
    maxMb: next,
    defaultMaxMb: DATA_URL_READ_DEFAULT_MAX_MB,
    maxBytes: dataUrlReadMaxBytesFromMb(next)
  };
});
ipcMain.handle("hermes:readFileDataUrl", async (_event, filePath) => {
  return readFileDataUrlForIpc(filePath, {
    maxBytes: dataUrlReadMaxBytesFromMb(dataUrlReadMaxMb),
    mimeType: mimeTypeForPath(resolveRequestedPathForIpc(filePath, { purpose: "File preview" })),
    purpose: "File preview"
  });
});
ipcMain.handle("hermes:readFileDataUrlForAttach", async (_event, filePath) => {
  return readFileDataUrlForIpc(filePath, {
    maxBytes: ATTACHMENT_UPLOAD_DEFAULT_MAX_BYTES,
    mimeType: mimeTypeForPath(resolveRequestedPathForIpc(filePath, { purpose: "Attachment upload" })),
    purpose: "Attachment upload"
  });
});
ipcMain.handle("hermes:readFileText", async (_event, filePath) => {
  const { resolvedPath, stat } = await resolveReadableFileForIpc(filePath, {
    maxBytes: TEXT_PREVIEW_SOURCE_MAX_BYTES,
    purpose: "Text preview"
  });
  const ext = path22.extname(resolvedPath).toLowerCase();
  const handle = await fs19.promises.open(resolvedPath, "r");
  const bytesToRead = Math.min(stat.size, TEXT_PREVIEW_MAX_BYTES);
  try {
    const buffer = Buffer.alloc(bytesToRead);
    const { bytesRead } = await handle.read(buffer, 0, bytesToRead, 0);
    return {
      binary: looksBinary(buffer.subarray(0, Math.min(bytesRead, 4096))),
      byteSize: stat.size,
      language: PREVIEW_LANGUAGE_BY_EXT[ext] || "text",
      mimeType: mimeTypeForPath(resolvedPath),
      path: resolvedPath,
      text: buffer.subarray(0, bytesRead).toString("utf8"),
      truncated: stat.size > TEXT_PREVIEW_MAX_BYTES
    };
  } finally {
    await handle.close();
  }
});
ipcMain.handle("hermes:selectPaths", async (_event, options = {}) => {
  const properties = options?.directories ? ["openDirectory"] : ["openFile"];
  if (options?.multiple !== false) {
    properties.push("multiSelections");
  }
  let resolvedDefaultPath;
  if (options?.defaultPath) {
    try {
      const bridged = IS_WINDOWS3 ? resolvePickerDefaultPath(String(options.defaultPath)) : String(options.defaultPath);
      resolvedDefaultPath = bridged ? path22.resolve(bridged) : void 0;
    } catch {
      resolvedDefaultPath = void 0;
    }
  }
  const result = await dialog.showOpenDialog(mainWindow, {
    title: options?.title || "Add context",
    defaultPath: resolvedDefaultPath,
    properties,
    filters: Array.isArray(options?.filters) ? options.filters : void 0
  });
  if (result.canceled) {
    return [];
  }
  return result.filePaths;
});
ipcMain.handle("hermes:writeClipboard", (_event, text) => {
  clipboard.writeText(String(text || ""));
  return true;
});
ipcMain.handle("hermes:readClipboard", () => clipboard.readText());
ipcMain.handle("hermes:saveImageFromUrl", (_event, url) => saveImageFromUrl(String(url || "")));
ipcMain.handle("hermes:saveImageBuffer", async (_event, payload) => {
  const data = payload?.data;
  if (!data) {
    throw new Error("saveImageBuffer: missing data");
  }
  const buffer = Buffer.isBuffer(data) ? data : Buffer.from(data);
  return writeComposerImage(buffer, payload?.ext || ".png");
});
ipcMain.handle("hermes:saveClipboardImage", async () => {
  const image = clipboard.readImage();
  if (image && !image.isEmpty()) {
    return writeComposerImage(image.toPNG(), ".png");
  }
  if (IS_WSL) {
    const png = readWslWindowsClipboardImage();
    if (png) {
      return writeComposerImage(png, ".png");
    }
  }
  return "";
});
ipcMain.handle(
  "hermes:normalizePreviewTarget",
  (_event, target2, baseDir) => normalizePreviewTarget(String(target2 || ""), baseDir ? String(baseDir) : "")
);
ipcMain.handle("hermes:watchPreviewFile", (_event, url) => watchPreviewFile(String(url || "")));
ipcMain.handle("hermes:watchDirectory", (_event, dir) => watchDirectory(String(dir || "")));
ipcMain.handle("hermes:stopPreviewFileWatch", (_event, id) => stopPreviewFileWatch(String(id || "")));
var activeWorkByWebContents = /* @__PURE__ */ new Map();
var streamThrottle = createStreamThrottle();
function updateStreamThrottleFromActiveWork() {
  streamThrottle.update(mergeActiveWork(activeWorkByWebContents.values()).count > 0);
}
ipcMain.on("hermes:active-work", (event, payload) => {
  const id = event.sender.id;
  if (!activeWorkByWebContents.has(id)) {
    event.sender.once("destroyed", () => {
      activeWorkByWebContents.delete(id);
      updateStreamThrottleFromActiveWork();
    });
  }
  activeWorkByWebContents.set(id, normalizeActiveWork(payload));
  updateStreamThrottleFromActiveWork();
});
ipcMain.on("hermes:titlebar-theme", (_event, payload) => {
  if (!payload || !isHexColor(payload.background) || !isHexColor(payload.foreground)) {
    return;
  }
  rendererTitleBarTheme = {
    background: payload.background,
    foreground: payload.foreground
  };
  for (const win of BrowserWindow2.getAllWindows()) {
    applyTitleBarOverlay(win);
  }
});
ipcMain.on("hermes:native-theme", (_event, mode) => {
  if (!THEME_SOURCES.has(mode)) {
    return;
  }
  if (nativeTheme.themeSource !== mode) {
    nativeTheme.themeSource = mode;
    writePersistedThemeSource(mode);
  }
});
ipcMain.on("hermes:translucency", (_event, payload) => {
  const next = clampIntensity(payload && payload.intensity);
  if (next === translucencyIntensity) {
    return;
  }
  translucencyIntensity = next;
  writePersistedTranslucency(next);
  for (const win of BrowserWindow2.getAllWindows()) {
    applyWindowTranslucency(win);
  }
});
var KEEP_AWAKE_CONFIG_PATH = path22.join(app2.getPath("userData"), "keep-awake.json");
var keepAwake = createKeepAwake(powerSaveBlocker);
function readPersistedKeepAwake() {
  try {
    return JSON.parse(fs19.readFileSync(KEEP_AWAKE_CONFIG_PATH, "utf8")).on === true;
  } catch {
    return false;
  }
}
ipcMain.on("hermes:keep-awake", (_event, on) => {
  const enabled2 = Boolean(on);
  keepAwake.set(enabled2);
  try {
    fs19.mkdirSync(path22.dirname(KEEP_AWAKE_CONFIG_PATH), { recursive: true });
    fs19.writeFileSync(KEEP_AWAKE_CONFIG_PATH, JSON.stringify({ on: enabled2 }, null, 2), "utf8");
  } catch (error) {
    rememberLog(`[keep-awake] write failed: ${error.message}`);
  }
});
ipcMain.handle("hermes:quick-entry:settings:get", async () => {
  const settings = readQuickEntrySettings();
  const state = quickEntryShortcut.current();
  return {
    enabled: settings.enabled,
    error: state.error,
    registered: state.registered,
    shortcut: settings.enabled ? state.shortcut : settings.shortcut
  };
});
ipcMain.handle("hermes:quick-entry:settings:set", async (_event, patch) => {
  const current = readQuickEntrySettings();
  const next = sanitizeQuickEntrySettings({
    enabled: patch?.enabled === void 0 ? current.enabled : patch.enabled === true,
    shortcut: typeof patch?.shortcut === "string" && patch.shortcut.trim() ? patch.shortcut : current.shortcut
  });
  writeQuickEntrySettings(next);
  return applyQuickEntrySettings(next);
});
ipcMain.on("hermes:quick-entry:submit", (_event, payload) => {
  hideQuickEntryWindow();
  const text = typeof payload?.text === "string" ? payload.text.trim() : "";
  if (!text) {
    return;
  }
  if (!mainWindow || mainWindow.isDestroyed()) {
    rememberLog("[quick-entry] dropped a submit: no primary window to route it to");
    return;
  }
  mainWindow.webContents.send("hermes:quick-entry:submit", {
    target: typeof payload?.target === "string" && payload.target ? payload.target : "current",
    text
  });
});
ipcMain.on("hermes:quick-entry:state", (_event, payload) => {
  quickEntryLastState = payload ?? null;
  if (quickEntryWindow && !quickEntryWindow.isDestroyed()) {
    quickEntryWindow.webContents.send("hermes:quick-entry:state", payload);
  }
});
ipcMain.on("hermes:quick-entry:dismiss", () => hideQuickEntryWindow());
ipcMain.handle("hermes:openExternal", (_event, url) => {
  if (!openExternalUrl(url)) {
    throw new Error("Invalid external URL");
  }
});
var foundInPageForwarders = /* @__PURE__ */ new Map();
function ensureFoundInPageForwarder(sender) {
  if (foundInPageForwarders.has(sender.id)) {
    return;
  }
  const uninstall = installFoundInPageForwarder(sender);
  foundInPageForwarders.set(sender.id, uninstall);
  sender.once("destroyed", () => {
    foundInPageForwarders.get(sender.id)?.();
    foundInPageForwarders.delete(sender.id);
  });
}
ipcMain.handle("hermes:find-in-page", (event, query, options) => {
  const win = BrowserWindow2.fromWebContents(event.sender);
  if (!win || win.isDestroyed()) {
    return { count: 0 };
  }
  ensureFoundInPageForwarder(event.sender);
  performFind(win.webContents, query, options);
  return { count: 0 };
});
ipcMain.handle("hermes:stop-find-in-page", (event) => {
  const win = BrowserWindow2.fromWebContents(event.sender);
  if (!win || win.isDestroyed()) {
    return;
  }
  stopFind(win.webContents);
});
ipcMain.handle("hermes:openPreviewInBrowser", async (_event, url) => {
  if (!await openPreviewInBrowser(url)) {
    throw new Error("Invalid preview URL");
  }
});
ipcMain.handle("hermes:setting:defaultProjectDir:get", async () => ({
  dir: readDefaultProjectDir(),
  defaultLabel: app2.getPath("home"),
  resolvedCwd: resolveHermesCwd()
}));
ipcMain.handle("hermes:workspace:sanitize", async (_event, cwd) => sanitizeWorkspaceCwd(cwd));
ipcMain.handle("hermes:setting:defaultProjectDir:set", async (_event, dir) => {
  const next = typeof dir === "string" && dir.trim() ? dir.trim() : null;
  if (next) {
    try {
      fs19.mkdirSync(next, { recursive: true });
    } catch (error) {
      throw new Error(`Could not create directory: ${error.message}`);
    }
  }
  writeDefaultProjectDir(next);
  return { dir: next };
});
ipcMain.handle("hermes:setting:defaultProjectDir:pick", async () => {
  const result = await dialog.showOpenDialog({
    title: "Choose default project directory",
    properties: ["openDirectory", "createDirectory"],
    defaultPath: readDefaultProjectDir() || app2.getPath("home")
  });
  if (result.canceled || result.filePaths.length === 0) {
    return { canceled: true, dir: null };
  }
  return { canceled: false, dir: result.filePaths[0] };
});
ipcMain.handle("hermes:fetchLinkTitle", (_event, url) => fetchLinkTitle(url));
ipcMain.handle("hermes:logs:reveal", async () => {
  try {
    await fs19.promises.mkdir(path22.dirname(DESKTOP_LOG_PATH), { recursive: true });
    if (!fileExists(DESKTOP_LOG_PATH)) {
      await fs19.promises.appendFile(DESKTOP_LOG_PATH, "");
    }
    shell.showItemInFolder(DESKTOP_LOG_PATH);
    return { ok: true, path: DESKTOP_LOG_PATH };
  } catch (error) {
    return { ok: false, path: DESKTOP_LOG_PATH, error: error.message };
  }
});
ipcMain.handle("hermes:logs:recent", async () => ({ path: DESKTOP_LOG_PATH, lines: hermesLog.slice(-200) }));
function isExecutableFile(filePath) {
  if (!filePath || !path22.isAbsolute(filePath)) {
    return false;
  }
  try {
    fs19.accessSync(filePath, fs19.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}
function posixShellSpec(shellPath) {
  const shellName = path22.basename(shellPath);
  const interactiveArgs = shellName.includes("zsh") || shellName.includes("bash") ? ["-il"] : ["-i"];
  return { args: interactiveArgs, command: shellPath, name: shellName };
}
function windowsPowerShellPath() {
  const systemRoot = process.env.SystemRoot || process.env.windir || "C:\\Windows";
  const builtin = path22.join(systemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  return isExecutableFile(builtin) ? builtin : findOnPath("powershell.exe");
}
function shellSpecFor(shellPath) {
  const name = path22.basename(shellPath).toLowerCase();
  if (name.startsWith("pwsh") || name.startsWith("powershell")) {
    return { args: ["-NoLogo"], command: shellPath, name };
  }
  if (name.startsWith("cmd")) {
    return { args: [], command: shellPath, name };
  }
  return posixShellSpec(shellPath);
}
function windowsShellSpec() {
  const command = findOnPath("pwsh.exe") || findOnPath("pwsh") || windowsPowerShellPath() || process.env.COMSPEC || "cmd.exe";
  return shellSpecFor(command);
}
function terminalShellCommand() {
  const override = (process.env.HERMES_DESKTOP_SHELL || (IS_WINDOWS3 ? "" : process.env.SHELL) || "").trim();
  if (override) {
    const resolved = isExecutableFile(override) ? override : findOnPath(override);
    if (resolved) {
      return shellSpecFor(resolved);
    }
  }
  if (IS_WINDOWS3) {
    return windowsShellSpec();
  }
  const shellPath = ["/bin/zsh", "/bin/bash", "/bin/sh"].find((candidate) => isExecutableFile(candidate));
  return posixShellSpec(shellPath || "/bin/sh");
}
function safeTerminalCwd(cwd) {
  const candidate = path22.resolve(String(cwd || app2.getPath("home")));
  try {
    const stat = fs19.statSync(candidate);
    return stat.isDirectory() ? candidate : path22.dirname(candidate);
  } catch {
    return app2.getPath("home");
  }
}
function terminalShellEnv() {
  const env2 = { ...process.env };
  for (const key of Object.keys(env2)) {
    if (key === "npm_config_prefix" || key.startsWith("npm_config_") || key.startsWith("npm_package_")) {
      delete env2[key];
    }
  }
  delete env2.NO_COLOR;
  delete env2.FORCE_COLOR;
  delete env2.COLORFGBG;
  env2.COLORTERM = "truecolor";
  env2.LC_CTYPE = env2.LC_CTYPE || "UTF-8";
  env2.TERM = "xterm-256color";
  env2.TERM_PROGRAM = "Hermes";
  env2.TERM_PROGRAM_VERSION = app2.getVersion();
  env2.HERMES_DESKTOP_TERMINAL = "1";
  return env2;
}
function terminalChannel(id, suffix) {
  return `hermes:terminal:${id}:${suffix}`;
}
function readProcessCwd(pid) {
  return new Promise((resolve) => {
    if (!Number.isInteger(pid) || pid <= 0) {
      resolve(null);
      return;
    }
    if (process.platform === "linux") {
      fs19.promises.readlink(`/proc/${pid}/cwd`).then((target2) => resolve(target2 || null)).catch(() => resolve(null));
      return;
    }
    if (process.platform === "darwin") {
      execFile4("lsof", ["-a", "-p", String(pid), "-d", "cwd", "-Fn"], { timeout: 2e3 }, (err, stdout) => {
        if (err) {
          resolve(null);
          return;
        }
        const line = String(stdout || "").split("\n").find((entry) => entry.startsWith("n"));
        resolve(line ? line.slice(1) : null);
      });
      return;
    }
    resolve(null);
  });
}
function disposeTerminalSession(id) {
  const sessionInfo = terminalSessions.get(id);
  if (!sessionInfo) {
    return false;
  }
  terminalSessions.delete(id);
  try {
    sessionInfo.pty.kill();
  } catch {
  }
  return true;
}
ipcMain.handle("hermes:fs:readDir", async (_event, dirPath) => readDirForIpc(dirPath));
ipcMain.handle("hermes:fs:gitRoot", async (_event, startPath) => gitRootForIpc(startPath));
ipcMain.handle("hermes:fs:reveal", async (_event, targetPath) => {
  const target2 = String(targetPath || "").trim();
  if (!target2) {
    return false;
  }
  try {
    shell.showItemInFolder(target2);
    return true;
  } catch {
    return false;
  }
});
ipcMain.handle("hermes:fs:openDir", async (_event, dirPath) => {
  const dir = String(dirPath || "").trim();
  if (!dir) {
    return { ok: false, error: "no path" };
  }
  try {
    await fs19.promises.mkdir(dir, { recursive: true });
    const error = await shell.openPath(path22.normalize(dir));
    return error ? { ok: false, error } : { ok: true };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
});
ipcMain.handle("hermes:fs:desktopPluginsRoot", async () => {
  const profile = readActiveDesktopProfile();
  const base = profile && profile !== "default" ? path22.join(HERMES_HOME, "profiles", profile) : HERMES_HOME;
  const dir = path22.join(base, "desktop-plugins");
  try {
    await fs19.promises.mkdir(dir, { recursive: true });
  } catch {
  }
  return dir;
});
ipcMain.handle("hermes:fs:rename", async (_event, targetPath, newName) => {
  const src = String(targetPath || "").trim();
  const name = String(newName || "").trim();
  if (!src || !name || name === "." || name === ".." || name.includes("/") || name.includes("\\")) {
    throw new Error("Invalid rename");
  }
  const dst = path22.join(path22.dirname(src), name);
  if (dst === src) {
    return { path: dst };
  }
  if (fs19.existsSync(dst)) {
    throw new Error(`"${name}" already exists`);
  }
  await fs19.promises.rename(src, dst);
  return { path: dst };
});
ipcMain.handle("hermes:fs:writeText", async (_event, filePath, content) => {
  const raw = String(filePath || "").trim();
  if (!raw) {
    throw new Error("Invalid path");
  }
  const text = String(content ?? "");
  if (text.length > 1e6) {
    throw new Error("Content too large");
  }
  const resolved = resolveRequestedPathForIpc(expandUserPath(raw), { purpose: "Write text file" });
  if (!directoryExists(path22.dirname(resolved))) {
    throw new Error("Parent directory does not exist");
  }
  await fs19.promises.writeFile(resolved, text, "utf8");
  return { path: resolved };
});
ipcMain.handle("hermes:fs:trash", async (_event, targetPath) => {
  const target2 = String(targetPath || "").trim();
  if (!target2) {
    throw new Error("Invalid delete");
  }
  await shell.trashItem(target2);
  return true;
});
ipcMain.handle("hermes:git:worktreeList", async (_event, repoPath) => listWorktrees(repoPath, resolveGitBinary()));
ipcMain.handle(
  "hermes:git:worktreeAdd",
  async (_event, repoPath, options) => addWorktree(repoPath, options || {}, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:worktreeRemove",
  async (_event, repoPath, worktreePath, options) => removeWorktree(repoPath, worktreePath, options || {}, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:branchSwitch",
  async (_event, repoPath, branch) => switchBranch(repoPath, branch, resolveGitBinary())
);
ipcMain.handle("hermes:git:branchList", async (_event, repoPath) => listBranches(repoPath, resolveGitBinary()));
ipcMain.handle("hermes:git:baseBranchList", async (_event, repoPath) => listBaseBranches(repoPath, resolveGitBinary()));
ipcMain.handle("hermes:git:repoStatus", async (_event, repoPath) => repoStatus(repoPath, resolveGitBinary()));
ipcMain.handle(
  "hermes:git:review:list",
  async (_event, repoPath, scope, baseRef) => reviewList(repoPath, scope, baseRef, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:diff",
  async (_event, repoPath, filePath, scope, baseRef, staged) => reviewDiff(repoPath, filePath, scope, baseRef, staged, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:fileDiff",
  async (_event, repoPath, filePath) => fileDiffVsHead(repoPath, filePath, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:stage",
  async (_event, repoPath, filePath) => reviewStage(repoPath, filePath ?? null, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:unstage",
  async (_event, repoPath, filePath) => reviewUnstage(repoPath, filePath ?? null, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:revert",
  async (_event, repoPath, filePath) => reviewRevert(repoPath, filePath ?? null, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:revParse",
  async (_event, repoPath, ref) => reviewRevParse(repoPath, ref, resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:commit",
  async (_event, repoPath, message, push) => reviewCommit(repoPath, message, Boolean(push), resolveGitBinary())
);
ipcMain.handle(
  "hermes:git:review:commitContext",
  async (_event, repoPath) => reviewCommitContext(repoPath, resolveGitBinary())
);
ipcMain.handle("hermes:git:review:push", async (_event, repoPath) => reviewPush(repoPath, resolveGitBinary()));
ipcMain.handle("hermes:git:review:shipInfo", async (_event, repoPath) => reviewShipInfo(repoPath, resolveGhBinary()));
ipcMain.handle(
  "hermes:git:review:createPr",
  async (_event, repoPath) => reviewCreatePr(repoPath, resolveGitBinary(), resolveGhBinary())
);
ipcMain.handle("hermes:git:scanRepos", async (_event, roots, options) => {
  try {
    return await scanGitRepos(roots || [], options || {});
  } catch {
    return [];
  }
});
var _spawnHelperEnsured = false;
function ensureNodePtySpawnHelper() {
  if (_spawnHelperEnsured || IS_WINDOWS3) {
    return;
  }
  _spawnHelperEnsured = true;
  try {
    const nodePtyRoot = path22.dirname(__require.resolve("node-pty/package.json"));
    const { fixed, errors } = ensureSpawnHelperExecutable(nodePtyRoot);
    for (const helperPath of fixed) {
      rememberLog(`[terminal] restored +x on node-pty spawn-helper: ${helperPath}`);
    }
    for (const failure of errors) {
      rememberLog(`[terminal] could not chmod spawn-helper ${failure.path}: ${failure.error}`);
    }
  } catch (error) {
    rememberLog(`[terminal] spawn-helper exec check skipped: ${error instanceof Error ? error.message : String(error)}`);
  }
}
ipcMain.handle("hermes:terminal:start", async (event, payload = {}) => {
  ensureNodePtySpawnHelper();
  const id = crypto6.randomUUID();
  const { args, command, name } = terminalShellCommand();
  const cwd = safeTerminalCwd(payload?.cwd);
  const cols = Math.max(2, Number.parseInt(String(payload?.cols || 80), 10) || 80);
  const rows = Math.max(2, Number.parseInt(String(payload?.rows || 24), 10) || 24);
  const sshTarget = await resolveTerminalConnection(activeSshTerminalTarget, () => ensureBackend(primaryProfileKey()));
  const remote = Boolean(sshTarget);
  const remoteState = remote ? sshConnections.get(sshTarget.scope) : null;
  const remoteCommand = remoteState?.remotePlatform === "Windows" ? buildWindowsInteractiveCommand(String(payload?.cwd || "").trim()) : void 0;
  const ptyProcess = remote ? nodePty.spawn(
    process.platform === "win32" ? path22.join(process.env.SystemRoot || "C:\\Windows", "System32", "OpenSSH", "ssh.exe") : "ssh",
    buildInteractiveSshArgs(sshTarget.ssh, String(payload?.cwd || "").trim(), void 0, remoteCommand),
    { cols, cwd: app2.getPath("home"), env: terminalShellEnv(), name: "xterm-256color", rows }
  ) : nodePty.spawn(command, args, { cols, cwd, env: terminalShellEnv(), name: "xterm-256color", rows });
  terminalSessions.set(id, {
    pty: ptyProcess,
    webContentsId: event.sender.id,
    ...remote ? { sshScope: sshTarget.scope, remoteCwd: String(payload?.cwd || "") } : {}
  });
  const send = (suffix, payload2) => {
    if (event.sender.isDestroyed()) {
      return;
    }
    event.sender.send(terminalChannel(id, suffix), payload2);
  };
  ptyProcess.onData((data) => send("data", data));
  ptyProcess.onExit(({ exitCode, signal }) => {
    terminalSessions.delete(id);
    send("exit", { code: exitCode, signal: signal || null });
  });
  event.sender.once("destroyed", () => disposeTerminalSession(id));
  return { cwd: remote ? null : cwd, id, shell: remote ? "ssh" : name };
});
ipcMain.handle("hermes:terminal:write", (_event, id, data) => {
  const sessionInfo = terminalSessions.get(String(id || ""));
  if (!sessionInfo) {
    return false;
  }
  sessionInfo.pty.write(String(data || ""));
  return true;
});
ipcMain.handle("hermes:terminal:resize", (_event, id, size = {}) => {
  const sessionInfo = terminalSessions.get(String(id || ""));
  if (!sessionInfo) {
    return false;
  }
  const cols = Math.max(2, Number.parseInt(String(size?.cols || 80), 10) || 80);
  const rows = Math.max(2, Number.parseInt(String(size?.rows || 24), 10) || 24);
  sessionInfo.pty.resize(cols, rows);
  return true;
});
ipcMain.handle("hermes:terminal:cwd", async (_event, id) => {
  const sessionInfo = terminalSessions.get(String(id || ""));
  if (!sessionInfo) {
    return null;
  }
  return sessionInfo.sshScope !== void 0 ? null : readProcessCwd(sessionInfo.pty.pid);
});
ipcMain.handle("hermes:terminal:dispose", (_event, id) => disposeTerminalSession(String(id || "")));
ipcMain.handle(
  "hermes:updates:check",
  async () => checkUpdates().catch((error) => ({
    supported: true,
    branch: readDesktopUpdateConfig().branch,
    error: "check-failed",
    message: error?.message || String(error),
    fetchedAt: Date.now()
  }))
);
ipcMain.handle(
  "hermes:updates:apply",
  async (_event, payload) => applyUpdates(payload || {}).catch((error) => ({
    ok: false,
    error: "apply-failed",
    message: error?.message || String(error)
  }))
);
ipcMain.handle("hermes:updates:branch:get", async () => readDesktopUpdateConfig());
ipcMain.handle("hermes:updates:branch:set", async (_event, name) => {
  const branch = typeof name === "string" && name.trim() ? name.trim() : DEFAULT_UPDATE_BRANCH;
  writeDesktopUpdateConfig({ branch });
  return { branch };
});
function resolveHermesVersion() {
  try {
    const root = resolveUpdateRoot();
    const initPath = path22.join(root, "hermes_cli", "__init__.py");
    if (fileExists(initPath)) {
      const raw = fs19.readFileSync(initPath, "utf8");
      const match = raw.match(/__version__\s*=\s*["']([^"']+)["']/);
      if (match) {
        return match[1];
      }
    }
  } catch {
  }
  return app2.getVersion();
}
function showAboutPanelFresh() {
  app2.setAboutPanelOptions({
    applicationName: APP_NAME,
    applicationVersion: resolveHermesVersion(),
    copyright: "Copyright \xA9 2026 Nous Research"
  });
  app2.showAboutPanel();
}
ipcMain.handle("hermes:version", async () => ({
  appVersion: resolveHermesVersion(),
  electronVersion: process.versions.electron,
  nodeVersion: process.versions.node,
  platform: process.platform,
  hermesRoot: resolveUpdateRoot()
}));
function uninstallVenvPython() {
  return getVenvPython(VENV_ROOT);
}
async function getUninstallSummary() {
  const py = uninstallVenvPython();
  const agentRoot = ACTIVE_HERMES_ROOT;
  const fallback = () => ({
    hermes_home: HERMES_HOME,
    agent_installed: isHermesSourceRoot(agentRoot) && fileExists(py),
    gui_installed: true,
    source_built_artifacts: [],
    packaged_app_paths: [],
    userdata_dir: app2.getPath("userData"),
    userdata_exists: true,
    platform: process.platform,
    probe: "fallback"
  });
  if (!fileExists(py)) {
    return fallback();
  }
  return new Promise((resolve) => {
    let stdout = "";
    let settled = false;
    const done = (value) => {
      if (settled) {
        return;
      }
      settled = true;
      resolve(value);
    };
    try {
      const child = spawn5(
        py,
        ["-m", "hermes_cli.main", "uninstall", "--gui-summary"],
        hiddenWindowsChildOptions({
          cwd: agentRoot,
          env: { ...process.env, HERMES_HOME, NO_COLOR: "1" },
          stdio: ["ignore", "pipe", "ignore"]
        })
      );
      child.stdout.on("data", (chunk) => {
        stdout += chunk.toString();
      });
      child.on("error", () => done(fallback()));
      child.on("exit", (code) => {
        if (code !== 0) {
          return done(fallback());
        }
        try {
          const line = stdout.trim().split("\n").filter(Boolean).pop() || "{}";
          const parsed = JSON.parse(line);
          parsed.running_app_path = resolveRemovableAppPath(process.execPath, process.platform, process.env);
          done(parsed);
        } catch {
          done(fallback());
        }
      });
      setTimeout(() => done(fallback()), 8e3);
    } catch {
      done(fallback());
    }
  });
}
async function runDesktopUninstall(mode) {
  let uninstallArgs;
  try {
    uninstallArgs = uninstallArgsForMode(mode);
  } catch (error) {
    return { ok: false, error: "invalid-mode", message: error.message };
  }
  const venvPy = uninstallVenvPython();
  if (!fileExists(venvPy)) {
    return {
      ok: false,
      error: "agent-missing",
      message: `Can't run the uninstaller: no Hermes agent venv at ${VENV_ROOT}.`
    };
  }
  let py = venvPy;
  let pythonPath = null;
  if (modeRemovesAgent(mode)) {
    const sysPy = findSystemPython();
    if (sysPy) {
      py = sysPy;
      pythonPath = ACTIVE_HERMES_ROOT;
    } else if (IS_WINDOWS3) {
      rememberLog(
        "[uninstall] no system Python found for lite/full on Windows; falling back to the venv python \u2014 venv files locked by the running interpreter may remain and need manual deletion."
      );
    }
  }
  const appPath = resolveRemovableAppPath(process.execPath, process.platform, process.env);
  const removeBundle = shouldRemoveAppBundle(IS_PACKAGED, appPath) ? appPath : null;
  try {
    await releaseBackendLock(ACTIVE_HERMES_ROOT, "uninstall");
  } catch (error) {
    rememberLog(`[uninstall] backend teardown errored (continuing): ${error.message}`);
  }
  const scriptArgs = {
    desktopPid: process.pid,
    pythonExe: py,
    pythonPath,
    agentRoot: ACTIVE_HERMES_ROOT,
    uninstallArgs,
    appPath: removeBundle,
    hermesHome: HERMES_HOME
  };
  let scriptPath;
  let runner;
  let runnerArgs;
  try {
    if (IS_WINDOWS3) {
      scriptPath = path22.join(app2.getPath("temp"), `hermes-uninstall-${Date.now()}.cmd`);
      fs19.writeFileSync(scriptPath, buildWindowsCleanupScript(scriptArgs));
      runner = process.env.ComSpec || "cmd.exe";
      runnerArgs = ["/c", scriptPath];
    } else {
      scriptPath = path22.join(app2.getPath("temp"), `hermes-uninstall-${Date.now()}.sh`);
      fs19.writeFileSync(scriptPath, buildPosixCleanupScript(scriptArgs), { mode: 493 });
      runner = "/bin/bash";
      runnerArgs = [scriptPath];
    }
  } catch (error) {
    return { ok: false, error: "script-write-failed", message: error.message };
  }
  try {
    const child = spawn5(runner, runnerArgs, {
      detached: true,
      stdio: "ignore",
      windowsHide: true
    });
    child.unref();
  } catch (error) {
    return { ok: false, error: "spawn-failed", message: error.message };
  }
  rememberLog(
    `[uninstall] launched detached cleanup (${mode}): ${scriptPath} (removesAgent=${modeRemovesAgent(mode)} removesUserData=${modeRemovesUserData(mode)} bundle=${removeBundle || "none"})`
  );
  isQuittingForHandoff = true;
  setTimeout(() => app2.quit(), 800);
  return { ok: true, mode, willRemoveAppBundle: Boolean(removeBundle), scriptPath };
}
ipcMain.handle("hermes:uninstall:summary", async () => getUninstallSummary());
ipcMain.handle("hermes:uninstall:run", async (_event, payload) => {
  const mode = payload && typeof payload === "object" ? payload.mode : payload;
  return runDesktopUninstall(String(mode || ""));
});
ipcMain.handle("hermes:vscode-theme:fetch", async (_event, id) => fetchMarketplaceThemes(String(id || "")));
ipcMain.handle("hermes:vscode-theme:search", async (_event, query) => searchMarketplaceThemes(String(query || ""), 20));
var HERMES_PROTOCOL = "hermes";
var _pendingDeepLink = null;
var _rendererReadyForDeepLink = false;
function _extractDeepLink(argv) {
  if (!Array.isArray(argv)) {
    return null;
  }
  return argv.find((a) => typeof a === "string" && a.startsWith(`${HERMES_PROTOCOL}://`)) || null;
}
function handleDeepLink(url) {
  if (!url || typeof url !== "string") {
    return;
  }
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    rememberLog(`[deeplink] ignoring malformed url: ${url}`);
    return;
  }
  const kind = parsed.hostname || "";
  const name = decodeURIComponent((parsed.pathname || "").replace(/^\//, ""));
  const params = {};
  parsed.searchParams.forEach((v, k2) => {
    params[k2] = v;
  });
  const payload = { kind, name, params };
  if (!_rendererReadyForDeepLink || !mainWindow || mainWindow.isDestroyed()) {
    _pendingDeepLink = payload;
    return;
  }
  try {
    if (mainWindow.isMinimized()) {
      mainWindow.restore();
    }
    mainWindow.focus();
    mainWindow.webContents.send("hermes:deep-link", payload);
    rememberLog(`[deeplink] delivered ${kind}/${name}`);
  } catch (err) {
    rememberLog(`[deeplink] delivery failed: ${err.message}`);
  }
}
ipcMain.handle("hermes:deep-link-ready", () => {
  _rendererReadyForDeepLink = true;
  if (_pendingDeepLink) {
    const queued = _pendingDeepLink;
    _pendingDeepLink = null;
    handleDeepLink(
      `${HERMES_PROTOCOL}://${queued.kind}/${encodeURIComponent(queued.name)}` + (Object.keys(queued.params).length ? "?" + new URLSearchParams(queued.params).toString() : "")
    );
  }
  return { ok: true };
});
function registerDeepLinkProtocol() {
  try {
    if (process.defaultApp && process.argv.length >= 2) {
      app2.setAsDefaultProtocolClient(HERMES_PROTOCOL, process.execPath, [path22.resolve(process.argv[1])]);
    } else {
      app2.setAsDefaultProtocolClient(HERMES_PROTOCOL);
    }
  } catch (err) {
    rememberLog(`[deeplink] protocol registration failed: ${err.message}`);
  }
}
var _gotSingleInstanceLock = app2.requestSingleInstanceLock();
if (!_gotSingleInstanceLock) {
  app2.quit();
} else {
  app2.on("second-instance", (_event, argv) => {
    const url = _extractDeepLink(argv);
    if (url) {
      handleDeepLink(url);
    }
    ensureMainWindow(mainWindow, {
      isReady: app2.isReady(),
      createWindow,
      focusWindow,
      // deep-link delivery focuses a live window after its renderer is ready.
      focusExisting: !url
    });
  });
}
app2.on("open-url", (event, url) => {
  event.preventDefault();
  handleDeepLink(url);
});
app2.whenReady().then(() => {
  const systemCa = installWindowsSystemCaTrust(tls);
  if (systemCa.applied) {
    rememberLog(
      `[tls] trusting ${systemCa.systemCertificateCount} Windows system CA certificate(s) for backend connections`
    );
  } else if (systemCa.error) {
    rememberLog(`[tls] could not load Windows system CA certificates: ${systemCa.error}`);
  }
  if (IS_MAC) {
    Menu2.setApplicationMenu(buildApplicationMenu());
  } else {
    Menu2.setApplicationMenu(null);
  }
  installMediaPermissions();
  registerMediaProtocol();
  installEmbedReferer();
  registerDeepLinkProtocol();
  ensureWslWindowsFonts();
  configureSpellChecker();
  registerPowerResumeListeners();
  keepAwake.set(readPersistedKeepAwake());
  applyQuickEntrySettings(readQuickEntrySettings());
  if (IS_MAC) {
    const reposition = () => wakeIndicatorController.reposition();
    screen2.on("display-added", reposition);
    screen2.on("display-metrics-changed", reposition);
    screen2.on("display-removed", reposition);
  }
  createWindow();
  const _coldStartLink = _extractDeepLink(process.argv);
  if (_coldStartLink) {
    handleDeepLink(_coldStartLink);
  }
  app2.on("activate", () => {
    if (!mainWindow || mainWindow.isDestroyed()) {
      createWindow();
    } else {
      focusWindow(mainWindow);
    }
  });
});
function configureSpellChecker() {
  try {
    const defaultSession = session2.defaultSession;
    if (!defaultSession || typeof defaultSession.setSpellCheckerLanguages !== "function") {
      return;
    }
    const available = defaultSession.availableSpellCheckerLanguages || [];
    const locale = app2.getLocale && app2.getLocale() || "en-US";
    const candidates = [locale, locale.split("-")[0], "en-US", "en"];
    const chosen = candidates.find((lang) => available.includes(lang)) || "en-US";
    defaultSession.setSpellCheckerLanguages([chosen]);
  } catch (error) {
    rememberLog(`Spellchecker setup failed: ${error.message}`);
  }
}
function heldQuitForActiveWork(event) {
  if (SKIP_QUIT_CONFIRM || quitConfirmedWithActiveWork || quitPromptOpen) {
    return false;
  }
  const prompt = quitPromptFor(mergeActiveWork(activeWorkByWebContents.values()), isQuittingForHandoff);
  const parent = BrowserWindow2.getFocusedWindow() ?? BrowserWindow2.getAllWindows()[0];
  if (!prompt || !parent || parent.isDestroyed()) {
    return false;
  }
  event.preventDefault();
  quitPromptOpen = true;
  void dialog.showMessageBox(parent, {
    buttons: ["Keep Running", "Quit Anyway"],
    cancelId: 0,
    defaultId: 0,
    detail: prompt.detail,
    message: prompt.message,
    type: "question"
  }).then(({ response }) => {
    quitPromptOpen = false;
    if (response === 1) {
      quitConfirmedWithActiveWork = true;
      app2.quit();
    }
  }).catch(() => {
    quitPromptOpen = false;
    quitConfirmedWithActiveWork = true;
    app2.quit();
  });
  return true;
}
app2.on("before-quit", (event) => {
  if (heldQuitForActiveWork(event)) {
    return;
  }
  isReallyQuitting = true;
  destroyTray();
  if ((sshConnections.size > 0 || sshBootstrapCoordinator.promises().length > 0) && !sshQuitTeardownDone) {
    event.preventDefault();
    sshBootstrapCoordinator.cancelAll();
    const scopes = [...sshConnections.keys()];
    const pending = Promise.allSettled([
      ...scopes.map((scope) => teardownSshConnection(scope || null)),
      ...sshBootstrapCoordinator.promises()
    ]);
    void Promise.race([pending, new Promise((resolve) => setTimeout(resolve, 4e3))]).then(async () => {
      await sshBootstrapCoordinator.forceCleanupAll();
      sshQuitTeardownDone = true;
      app2.quit();
    });
  }
  if (IS_WINDOWS3 && !windowsSandboxFallbackSticky) {
    try {
      writeSandboxMarker(app2.getPath("userData"), markerAfterSuccessfulBoot({ fallbackActive: false }));
    } catch {
    }
  }
  closePetOverlay();
  wakeIndicatorController.close();
  closeQuickEntryWindow();
  if (bootstrapAbortController) {
    try {
      bootstrapAbortController.abort();
    } catch {
    }
  }
  if (desktopLogFlushTimer) {
    clearTimeout(desktopLogFlushTimer);
    desktopLogFlushTimer = null;
  }
  flushDesktopLogBufferSync();
  closePreviewWatchers();
  for (const id of [...terminalSessions.keys()]) {
    disposeTerminalSession(id);
  }
  stopBackendChild2(backendConnectionState.getProcess());
  stopAllPoolBackends();
});
app2.on("window-all-closed", () => {
  if (process.platform !== "darwin" || isQuittingForHandoff) {
    app2.quit();
  }
});
