(function (root, factory) {
  "use strict";
  if (typeof module === "object" && module.exports) {
    module.exports = factory(null);
  } else {
    factory(root);
  }
})(typeof window !== "undefined" ? window : null, function (root) {
  "use strict";

  const PLUGIN_NAME = "buzz-platform";
  const POLICY_URL = "/api/plugins/buzz-platform/policy";
  const DEFAULT_POLICY = Object.freeze({
    allowed_users: [],
    allow_all_users: false,
    require_mention: true,
    thread_require_mention: true,
  });
  const BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : {};
  }

  function splitAllowedUsers(raw) {
    return String(raw || "")
      .split(/[\n,]+/)
      .map(function (item) { return item.trim(); })
      .filter(Boolean);
  }

  function bech32Polymod(values) {
    const generators = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3];
    let checksum = 1;
    values.forEach(function (value) {
      const top = checksum >>> 25;
      checksum = ((checksum & 0x1ffffff) << 5) ^ value;
      generators.forEach(function (generator, bit) {
        if ((top >>> bit) & 1) checksum ^= generator;
      });
    });
    return checksum;
  }

  function validNpub(value) {
    const normalized = String(value || "").trim().toLowerCase();
    if (!normalized.startsWith("npub1")) return false;
    const data = normalized.slice(5).split("").map(function (character) {
      return BECH32_CHARSET.indexOf(character);
    });
    if (data.length < 7 || data.some(function (part) { return part < 0; })) return false;
    const hrp = "npub".split("");
    const expandedHrp = hrp.map(function (character) { return character.charCodeAt(0) >>> 5; })
      .concat([0])
      .concat(hrp.map(function (character) { return character.charCodeAt(0) & 31; }));
    if (bech32Polymod(expandedHrp.concat(data)) !== 1) return false;

    let accumulator = 0;
    let bits = 0;
    const decoded = [];
    data.slice(0, -6).forEach(function (part) {
      accumulator = (accumulator << 5) | part;
      bits += 5;
      while (bits >= 8) {
        bits -= 8;
        decoded.push((accumulator >>> bits) & 0xff);
      }
    });
    return decoded.length === 32 && bits < 5 && ((accumulator << (8 - bits)) & 0xff) === 0;
  }

  function validateAllowedUsers(raw) {
    const entries = splitAllowedUsers(raw);
    for (let index = 0; index < entries.length; index += 1) {
      const item = entries[index];
      if (/\s/.test(item)) {
        return "Invalid public identity at item " + (index + 1) + ". Put one identity on each line.";
      }
      if (!/^[0-9a-f]{64}$/i.test(item) && !validNpub(item)) {
        return "Invalid public identity at item " + (index + 1) +
          "; use a checksum-valid npub or 64-character hex public key.";
      }
    }
    return null;
  }

  function updateAllowedUsersDraft(raw) {
    const draft = String(raw == null ? "" : raw);
    return { draft: draft, error: validateAllowedUsers(draft) };
  }

  function settingsFromPayload(payload) {
    const response = asObject(payload);
    const policy = asObject(response.policy);
    const indeterminateFields = Array.isArray(response.indeterminate_fields)
      ? response.indeterminate_fields.slice() : [];
    const settings = { indeterminateFields: indeterminateFields };
    if (!indeterminateFields.includes("allowed_users")) {
      settings.allowedUsersText = Array.isArray(policy.allowed_users)
        ? policy.allowed_users.join("\n") : "";
    }
    if (!indeterminateFields.includes("allow_all_users")) {
      settings.allowAllUsers = Boolean(policy.allow_all_users);
    }
    if (!indeterminateFields.includes("require_mention")) {
      settings.requireMention = Boolean(policy.require_mention);
    }
    if (!indeterminateFields.includes("thread_require_mention")) {
      settings.threadRequireMention = Boolean(policy.thread_require_mention);
    }
    return settings;
  }

  function buildPolicyBody(settings) {
    const policy = {};
    if (Object.prototype.hasOwnProperty.call(settings, "allowedUsersText")) {
      policy.allowed_users = splitAllowedUsers(settings.allowedUsersText);
    }
    if (Object.prototype.hasOwnProperty.call(settings, "allowAllUsers")) {
      policy.allow_all_users = Boolean(settings.allowAllUsers);
    }
    if (Object.prototype.hasOwnProperty.call(settings, "requireMention")) {
      policy.require_mention = Boolean(settings.requireMention);
    }
    if (Object.prototype.hasOwnProperty.call(settings, "threadRequireMention")) {
      policy.thread_require_mention = Boolean(settings.threadRequireMention);
    }
    return { policy: policy };
  }

  function policyURL(profile) {
    return profile ? POLICY_URL + "?profile=" + encodeURIComponent(profile) : POLICY_URL;
  }

  const helpers = {
    splitAllowedUsers: splitAllowedUsers,
    validNpub: validNpub,
    validateAllowedUsers: validateAllowedUsers,
    updateAllowedUsersDraft: updateAllowedUsersDraft,
    settingsFromPayload: settingsFromPayload,
    buildPolicyBody: buildPolicyBody,
    policyURL: policyURL,
  };

  if (!root) return helpers;

  const SDK = root.__HERMES_PLUGIN_SDK__;
  const registry = root.__HERMES_PLUGINS__;
  if (!SDK || !registry || typeof SDK.fetchJSON !== "function") {
    throw new Error("Buzz requires the Hermes Dashboard plugin SDK");
  }

  const React = SDK.React;
  const h = React.createElement;
  const useState = SDK.hooks.useState;
  const useEffect = SDK.hooks.useEffect;
  const useRef = SDK.hooks.useRef;

  function selectedProfile() {
    try {
      if (SDK.api && typeof SDK.api.getManagementProfile === "function") {
        return String(SDK.api.getManagementProfile() || "").trim();
      }
    } catch (_error) {
      return "";
    }
    return "";
  }

  function FieldStatus(props) {
    if (!props.overridden) return null;
    return h("span", { className: "buzz-policy-override", role: "status" },
      "Active environment value is hidden. The configured value underneath is not shown or used until the override is removed.",
    );
  }

  function BuzzSectionIcon(props) {
    const className = ["buzz-config-section-icon", props && props.className]
      .filter(Boolean)
      .join(" ");
    return h("span", { className: className, "aria-hidden": "true" });
  }

  function SwitchRow(props) {
    return h("label", { className: "buzz-policy-switch" },
      h("input", {
        type: "checkbox",
        checked: props.checked,
        ref: function (element) { if (element) element.indeterminate = Boolean(props.indeterminate); },
        onChange: function (event) { props.onChange(Boolean(event.target.checked)); },
        disabled: props.disabled,
        "aria-label": props.label,
        "aria-checked": props.indeterminate ? "mixed" : props.checked,
      }),
      h("span", { className: "buzz-policy-switch-copy" },
        h("strong", null, props.label),
        h("span", { className: "buzz-policy-help" }, props.help),
        h(FieldStatus, { overridden: props.overridden }),
      ),
    );
  }

  function BuzzPolicyPanel() {
    const profile = selectedProfile();
    const settingsPair = useState(settingsFromPayload({ policy: DEFAULT_POLICY }));
    const settings = settingsPair[0];
    const setSettings = settingsPair[1];
    const metadataPair = useState({
      environment_overrides: [], indeterminate_fields: [], managed_fields: [], locked: false,
      managed_error: false, legacy_cleanup_required: false,
      additional_global_grants_active: false, additional_pairing_grants_active: false,
    });
    const metadata = metadataPair[0];
    const setMetadata = metadataPair[1];
    const loadingPair = useState(true);
    const loading = loadingPair[0];
    const setLoading = loadingPair[1];
    const readyPair = useState(false);
    const ready = readyPair[0];
    const setReady = readyPair[1];
    const savingPair = useState(false);
    const saving = savingPair[0];
    const setSaving = savingPair[1];
    const errorPair = useState("");
    const error = errorPair[0];
    const setError = errorPair[1];
    const statusPair = useState("");
    const status = statusPair[0];
    const setStatus = statusPair[1];
    const validationPair = useState("");
    const validation = validationPair[0];
    const setValidation = validationPair[1];
    const legacyResolvedPair = useState(false);
    const legacyResolved = legacyResolvedPair[0];
    const setLegacyResolved = legacyResolvedPair[1];
    const requestSequence = useRef(0);

    useEffect(function () {
      const sequence = requestSequence.current + 1;
      requestSequence.current = sequence;
      setLoading(true);
      setReady(false);
      setSaving(false);
      setError("");
      setStatus("");
      setValidation("");
      setLegacyResolved(false);
      SDK.fetchJSON(policyURL(profile))
        .then(function (payload) {
          if (sequence !== requestSequence.current) return;
          setSettings(settingsFromPayload(payload));
          setMetadata(asObject(payload));
          setReady(true);
        })
        .catch(function () {
          if (sequence !== requestSequence.current) return;
          setError("Could not load Buzz policy for the selected profile.");
        })
        .finally(function () {
          if (sequence !== requestSequence.current) return;
          setLoading(false);
        });
      return function () {
        if (sequence === requestSequence.current) requestSequence.current += 1;
      };
    }, [profile]);

    function change(key, value) {
      setSettings(Object.assign({}, settings, { [key]: value }));
      setStatus("");
      setError("");
    }

    function changeAllowedUsers(raw) {
      const next = updateAllowedUsersDraft(raw);
      change("allowedUsersText", next.draft);
      setValidation(next.error || "");
    }

    function save(event) {
      if (event && typeof event.preventDefault === "function") event.preventDefault();
      const validationError = Object.prototype.hasOwnProperty.call(settings, "allowedUsersText")
        ? validateAllowedUsers(settings.allowedUsersText) : null;
      setValidation(validationError || "");
      if (validationError || loading || saving || !ready || metadata.locked) return;

      const sequence = requestSequence.current + 1;
      requestSequence.current = sequence;
      const hadLegacy = Boolean(metadata.legacy_cleanup_required);
      setSaving(true);
      setError("");
      setStatus("");
      SDK.fetchJSON(policyURL(profile), {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildPolicyBody(settings)),
      })
        .then(function (payload) {
          if (sequence !== requestSequence.current) return;
          setSettings(settingsFromPayload(payload));
          setMetadata(asObject(payload));
          setLegacyResolved(hadLegacy && !payload.legacy_cleanup_required);
          const savedOverrides = Array.isArray(payload.environment_overrides)
            ? payload.environment_overrides : [];
          setStatus(savedOverrides.length
            ? "Saved. Non-overridden policy changes apply immediately; overridden fields remain ineffective until their environment override is removed. No Gateway restart is required."
            : "Saved. Policy changes apply immediately. No Gateway restart is required.");
        })
        .catch(function (saveError) {
          if (sequence !== requestSequence.current) return;
          if (String(saveError && saveError.message).includes("managed_")) {
            setMetadata(Object.assign({}, metadata, { locked: true }));
            setError("Save refused because this profile now has a managed policy.");
          } else {
            setError("Could not save Buzz policy. Your draft is still here; review it and try again.");
          }
        })
        .finally(function () {
          if (sequence !== requestSequence.current) return;
          setSaving(false);
        });
    }

    const overrides = Array.isArray(metadata.environment_overrides)
      ? metadata.environment_overrides : [];
    const managedFields = Array.isArray(metadata.managed_fields) ? metadata.managed_fields : [];
    const indeterminateFields = Array.isArray(metadata.indeterminate_fields)
      ? metadata.indeterminate_fields : [];
    const locked = Boolean(metadata.locked);
    const disabled = loading || saving || !ready || locked;
    const hasEditablePolicyField = [
      "allowed_users", "allow_all_users", "require_mention", "thread_require_mention",
    ].some(function (field) { return !indeterminateFields.includes(field); });
    function fieldDisabled(field) {
      return disabled || indeterminateFields.includes(field);
    }

    return h("section", {
      className: "buzz-policy-section",
      "aria-labelledby": "buzz-policy-title",
      "aria-busy": loading || saving,
    },
      h("header", { className: "buzz-policy-hero" },
        h("span", { className: "buzz-policy-logo", role: "img", "aria-label": "Buzz" }),
        h("div", null,
          h("p", { className: "buzz-policy-kicker" }, "Platform policy"),
          h("h2", { id: "buzz-policy-title" }, "Buzz-specific policy"),
          h("p", { className: "buzz-policy-subtitle" },
            "Configure Buzz-specific access and mention behavior for this Hermes profile.",
          ),
        ),
      ),
      h("p", { className: "buzz-policy-live", role: "note" },
        overrides.length
          ? "Non-overridden policy changes apply immediately. Overridden fields remain ineffective until their environment override is removed. No Gateway restart is required."
          : "Policy changes apply immediately. No Gateway restart is required.",
      ),
      h("p", { className: "buzz-policy-warning", role: "note" },
        metadata.additional_global_grants_active || metadata.additional_pairing_grants_active
          ? "Additional access is active. Gateway or pairing grants can broaden access beyond the Buzz-specific policy shown here; grant identities remain private."
          : "Gateway or pairing grants can broaden access beyond this Buzz-specific policy. Pairing identities are never shown here.",
      ),
      locked ? h("div", { className: "buzz-policy-lock", role: "alert" },
        h("strong", null, metadata.user_policy_unavailable
          ? "User policy unavailable" : metadata.managed_error
            ? "Managed policy unavailable" : "Locked by managed policy"),
        h("span", null, metadata.user_policy_unavailable
          ? "This panel is read-only because the user policy could not be inspected safely."
          : metadata.managed_error
            ? "This panel is read-only because the managed policy could not be inspected safely."
          : managedFields.length
            ? "This panel is read-only because these managed fields control it: " + managedFields.join(", ") + "."
            : "This panel is read-only because a managed policy controls it."),
      ) : null,
      metadata.legacy_cleanup_required ? h("p", { className: "buzz-policy-warning", role: "status" },
        "Legacy policy keys will be cleaned up when you save.",
      ) : null,
      legacyResolved ? h("p", { className: "buzz-policy-success", role: "status" },
        "Legacy policy keys were cleaned up by this save.",
      ) : null,
      h("form", { onSubmit: save },
        h("div", { className: "buzz-policy-field" },
          h("label", { htmlFor: "buzz-policy-allowed-users" }, "Allowed identities"),
          h("textarea", {
            id: "buzz-policy-allowed-users",
            value: settings.allowedUsersText || "",
            onChange: function (event) { changeAllowedUsers(event.target.value); },
            disabled: fieldDisabled("allowed_users"),
            placeholder: indeterminateFields.includes("allowed_users")
              ? "Active environment identities are hidden"
              : "One checksum-valid npub or 64-character hex key per line",
            rows: 5,
            spellCheck: false,
            autoCapitalize: "none",
            autoCorrect: "off",
            "aria-invalid": Boolean(validation),
            "aria-describedby": validation
              ? "buzz-policy-allowed-help buzz-policy-allowed-error"
              : "buzz-policy-allowed-help",
          }),
          h("p", { id: "buzz-policy-allowed-help", className: "buzz-policy-help" },
            "One public identity per line. Commas are also accepted. Invalid text stays in this draft until you correct it.",
          ),
          validation ? h("p", {
            id: "buzz-policy-allowed-error", className: "buzz-policy-error", role: "alert",
          }, validation) : null,
          h(FieldStatus, { overridden: overrides.includes("allowed_users") }),
        ),
        h("div", { className: "buzz-policy-switches" },
          h(SwitchRow, {
            label: "Allow all",
            checked: settings.allowAllUsers,
            onChange: function (value) { change("allowAllUsers", value); },
            disabled: fieldDisabled("allow_all_users"),
            overridden: overrides.includes("allow_all_users"),
            indeterminate: indeterminateFields.includes("allow_all_users"),
            help: "Allow any member of the connected Buzz community to instruct Hermes.",
          }),
          h(SwitchRow, {
            label: "Require mention",
            checked: settings.requireMention,
            onChange: function (value) { change("requireMention", value); },
            disabled: fieldDisabled("require_mention"),
            overridden: overrides.includes("require_mention"),
            indeterminate: indeterminateFields.includes("require_mention"),
            help: "In shared channels, respond only when this Hermes agent is mentioned.",
          }),
          h(SwitchRow, {
            label: "Require mention in threaded replies",
            checked: settings.threadRequireMention,
            onChange: function (value) { change("threadRequireMention", value); },
            disabled: fieldDisabled("thread_require_mention"),
            overridden: overrides.includes("thread_require_mention"),
            indeterminate: indeterminateFields.includes("thread_require_mention"),
            help: "Keep follow-up replies gated inside active threads unless Hermes is mentioned.",
          }),
        ),
        error ? h("p", { className: "buzz-policy-error", role: "alert" }, error) : null,
        h("div", { className: "buzz-policy-actions" },
          h("button", {
            className: "buzz-policy-save", type: "submit",
            disabled: disabled || Boolean(validation) || !hasEditablePolicyField,
          }, loading ? "Loading…" : saving ? "Saving…" : "Save Buzz policy"),
          h("p", { className: "buzz-policy-status", role: "status", "aria-live": "polite" }, status),
        ),
      ),
    );
  }

  registry.registerSlot(
    PLUGIN_NAME,
    "config:section:buzz",
    BuzzPolicyPanel,
    { icon: BuzzSectionIcon, optionCount: 4 },
  );
  return helpers;
});
