//#region ../../node_modules/.pnpm/@lit+reactive-element@2.1.2/node_modules/@lit/reactive-element/css-tag.js
/**
* @license
* Copyright 2019 Google LLC
* SPDX-License-Identifier: BSD-3-Clause
*/
const t$4 = globalThis, e$6 = t$4.ShadowRoot && (void 0 === t$4.ShadyCSS || t$4.ShadyCSS.nativeShadow) && "adoptedStyleSheets" in Document.prototype && "replace" in CSSStyleSheet.prototype, s$4 = Symbol(), o$6 = /* @__PURE__ */ new WeakMap();
var n$5 = class {
	constructor(t, e, o) {
		if (this._$cssResult$ = !0, o !== s$4) throw Error("CSSResult is not constructable. Use `unsafeCSS` or `css` instead.");
		this.cssText = t, this.t = e;
	}
	get styleSheet() {
		let t = this.o;
		const s = this.t;
		if (e$6 && void 0 === t) {
			const e = void 0 !== s && 1 === s.length;
			e && (t = o$6.get(s)), void 0 === t && ((this.o = t = new CSSStyleSheet()).replaceSync(this.cssText), e && o$6.set(s, t));
		}
		return t;
	}
	toString() {
		return this.cssText;
	}
};
const r$4 = (t) => new n$5("string" == typeof t ? t : t + "", void 0, s$4), S$1 = (s, o) => {
	if (e$6) s.adoptedStyleSheets = o.map((t) => t instanceof CSSStyleSheet ? t : t.styleSheet);
	else for (const e of o) {
		const o = document.createElement("style"), n = t$4.litNonce;
		void 0 !== n && o.setAttribute("nonce", n), o.textContent = e.cssText, s.appendChild(o);
	}
}, c$4 = e$6 ? (t) => t : (t) => t instanceof CSSStyleSheet ? ((t) => {
	let e = "";
	for (const s of t.cssRules) e += s.cssText;
	return r$4(e);
})(t) : t;
//#endregion
//#region ../../node_modules/.pnpm/@lit+reactive-element@2.1.2/node_modules/@lit/reactive-element/reactive-element.js
/**
* @license
* Copyright 2017 Google LLC
* SPDX-License-Identifier: BSD-3-Clause
*/ const { is: i$4, defineProperty: e$5, getOwnPropertyDescriptor: h$4, getOwnPropertyNames: r$3, getOwnPropertySymbols: o$5, getPrototypeOf: n$4 } = Object, a$1 = globalThis, c$3 = a$1.trustedTypes, l$2 = c$3 ? c$3.emptyScript : "", p$2 = a$1.reactiveElementPolyfillSupport, d$2 = (t, s) => t, u$2 = {
	toAttribute(t, s) {
		switch (s) {
			case Boolean:
				t = t ? l$2 : null;
				break;
			case Object:
			case Array: t = null == t ? t : JSON.stringify(t);
		}
		return t;
	},
	fromAttribute(t, s) {
		let i = t;
		switch (s) {
			case Boolean:
				i = null !== t;
				break;
			case Number:
				i = null === t ? null : Number(t);
				break;
			case Object:
			case Array: try {
				i = JSON.parse(t);
			} catch (t) {
				i = null;
			}
		}
		return i;
	}
}, f$3 = (t, s) => !i$4(t, s), b$1 = {
	attribute: !0,
	type: String,
	converter: u$2,
	reflect: !1,
	useDefault: !1,
	hasChanged: f$3
};
Symbol.metadata ??= Symbol("metadata"), a$1.litPropertyMetadata ??= /* @__PURE__ */ new WeakMap();
var y$1 = class extends HTMLElement {
	static addInitializer(t) {
		this._$Ei(), (this.l ??= []).push(t);
	}
	static get observedAttributes() {
		return this.finalize(), this._$Eh && [...this._$Eh.keys()];
	}
	static createProperty(t, s = b$1) {
		if (s.state && (s.attribute = !1), this._$Ei(), this.prototype.hasOwnProperty(t) && ((s = Object.create(s)).wrapped = !0), this.elementProperties.set(t, s), !s.noAccessor) {
			const i = Symbol(), h = this.getPropertyDescriptor(t, i, s);
			void 0 !== h && e$5(this.prototype, t, h);
		}
	}
	static getPropertyDescriptor(t, s, i) {
		const { get: e, set: r } = h$4(this.prototype, t) ?? {
			get() {
				return this[s];
			},
			set(t) {
				this[s] = t;
			}
		};
		return {
			get: e,
			set(s) {
				const h = e?.call(this);
				r?.call(this, s), this.requestUpdate(t, h, i);
			},
			configurable: !0,
			enumerable: !0
		};
	}
	static getPropertyOptions(t) {
		return this.elementProperties.get(t) ?? b$1;
	}
	static _$Ei() {
		if (this.hasOwnProperty(d$2("elementProperties"))) return;
		const t = n$4(this);
		t.finalize(), void 0 !== t.l && (this.l = [...t.l]), this.elementProperties = new Map(t.elementProperties);
	}
	static finalize() {
		if (this.hasOwnProperty(d$2("finalized"))) return;
		if (this.finalized = !0, this._$Ei(), this.hasOwnProperty(d$2("properties"))) {
			const t = this.properties, s = [...r$3(t), ...o$5(t)];
			for (const i of s) this.createProperty(i, t[i]);
		}
		const t = this[Symbol.metadata];
		if (null !== t) {
			const s = litPropertyMetadata.get(t);
			if (void 0 !== s) for (const [t, i] of s) this.elementProperties.set(t, i);
		}
		this._$Eh = /* @__PURE__ */ new Map();
		for (const [t, s] of this.elementProperties) {
			const i = this._$Eu(t, s);
			void 0 !== i && this._$Eh.set(i, t);
		}
		this.elementStyles = this.finalizeStyles(this.styles);
	}
	static finalizeStyles(s) {
		const i = [];
		if (Array.isArray(s)) {
			const e = new Set(s.flat(Infinity).reverse());
			for (const s of e) i.unshift(c$4(s));
		} else void 0 !== s && i.push(c$4(s));
		return i;
	}
	static _$Eu(t, s) {
		const i = s.attribute;
		return !1 === i ? void 0 : "string" == typeof i ? i : "string" == typeof t ? t.toLowerCase() : void 0;
	}
	constructor() {
		super(), this._$Ep = void 0, this.isUpdatePending = !1, this.hasUpdated = !1, this._$Em = null, this._$Ev();
	}
	_$Ev() {
		this._$ES = new Promise((t) => this.enableUpdating = t), this._$AL = /* @__PURE__ */ new Map(), this._$E_(), this.requestUpdate(), this.constructor.l?.forEach((t) => t(this));
	}
	addController(t) {
		(this._$EO ??= /* @__PURE__ */ new Set()).add(t), void 0 !== this.renderRoot && this.isConnected && t.hostConnected?.();
	}
	removeController(t) {
		this._$EO?.delete(t);
	}
	_$E_() {
		const t = /* @__PURE__ */ new Map(), s = this.constructor.elementProperties;
		for (const i of s.keys()) this.hasOwnProperty(i) && (t.set(i, this[i]), delete this[i]);
		t.size > 0 && (this._$Ep = t);
	}
	createRenderRoot() {
		const t = this.shadowRoot ?? this.attachShadow(this.constructor.shadowRootOptions);
		return S$1(t, this.constructor.elementStyles), t;
	}
	connectedCallback() {
		this.renderRoot ??= this.createRenderRoot(), this.enableUpdating(!0), this._$EO?.forEach((t) => t.hostConnected?.());
	}
	enableUpdating(t) {}
	disconnectedCallback() {
		this._$EO?.forEach((t) => t.hostDisconnected?.());
	}
	attributeChangedCallback(t, s, i) {
		this._$AK(t, i);
	}
	_$ET(t, s) {
		const i = this.constructor.elementProperties.get(t), e = this.constructor._$Eu(t, i);
		if (void 0 !== e && !0 === i.reflect) {
			const h = (void 0 !== i.converter?.toAttribute ? i.converter : u$2).toAttribute(s, i.type);
			this._$Em = t, null == h ? this.removeAttribute(e) : this.setAttribute(e, h), this._$Em = null;
		}
	}
	_$AK(t, s) {
		const i = this.constructor, e = i._$Eh.get(t);
		if (void 0 !== e && this._$Em !== e) {
			const t = i.getPropertyOptions(e), h = "function" == typeof t.converter ? { fromAttribute: t.converter } : void 0 !== t.converter?.fromAttribute ? t.converter : u$2;
			this._$Em = e;
			const r = h.fromAttribute(s, t.type);
			this[e] = r ?? this._$Ej?.get(e) ?? r, this._$Em = null;
		}
	}
	requestUpdate(t, s, i, e = !1, h) {
		if (void 0 !== t) {
			const r = this.constructor;
			if (!1 === e && (h = this[t]), i ??= r.getPropertyOptions(t), !((i.hasChanged ?? f$3)(h, s) || i.useDefault && i.reflect && h === this._$Ej?.get(t) && !this.hasAttribute(r._$Eu(t, i)))) return;
			this.C(t, s, i);
		}
		!1 === this.isUpdatePending && (this._$ES = this._$EP());
	}
	C(t, s, { useDefault: i, reflect: e, wrapped: h }, r) {
		i && !(this._$Ej ??= /* @__PURE__ */ new Map()).has(t) && (this._$Ej.set(t, r ?? s ?? this[t]), !0 !== h || void 0 !== r) || (this._$AL.has(t) || (this.hasUpdated || i || (s = void 0), this._$AL.set(t, s)), !0 === e && this._$Em !== t && (this._$Eq ??= /* @__PURE__ */ new Set()).add(t));
	}
	async _$EP() {
		this.isUpdatePending = !0;
		try {
			await this._$ES;
		} catch (t) {
			Promise.reject(t);
		}
		const t = this.scheduleUpdate();
		return null != t && await t, !this.isUpdatePending;
	}
	scheduleUpdate() {
		return this.performUpdate();
	}
	performUpdate() {
		if (!this.isUpdatePending) return;
		if (!this.hasUpdated) {
			if (this.renderRoot ??= this.createRenderRoot(), this._$Ep) {
				for (const [t, s] of this._$Ep) this[t] = s;
				this._$Ep = void 0;
			}
			const t = this.constructor.elementProperties;
			if (t.size > 0) for (const [s, i] of t) {
				const { wrapped: t } = i, e = this[s];
				!0 !== t || this._$AL.has(s) || void 0 === e || this.C(s, void 0, i, e);
			}
		}
		let t = !1;
		const s = this._$AL;
		try {
			t = this.shouldUpdate(s), t ? (this.willUpdate(s), this._$EO?.forEach((t) => t.hostUpdate?.()), this.update(s)) : this._$EM();
		} catch (s) {
			throw t = !1, this._$EM(), s;
		}
		t && this._$AE(s);
	}
	willUpdate(t) {}
	_$AE(t) {
		this._$EO?.forEach((t) => t.hostUpdated?.()), this.hasUpdated || (this.hasUpdated = !0, this.firstUpdated(t)), this.updated(t);
	}
	_$EM() {
		this._$AL = /* @__PURE__ */ new Map(), this.isUpdatePending = !1;
	}
	get updateComplete() {
		return this.getUpdateComplete();
	}
	getUpdateComplete() {
		return this._$ES;
	}
	shouldUpdate(t) {
		return !0;
	}
	update(t) {
		this._$Eq &&= this._$Eq.forEach((t) => this._$ET(t, this[t])), this._$EM();
	}
	updated(t) {}
	firstUpdated(t) {}
};
y$1.elementStyles = [], y$1.shadowRootOptions = { mode: "open" }, y$1[d$2("elementProperties")] = /* @__PURE__ */ new Map(), y$1[d$2("finalized")] = /* @__PURE__ */ new Map(), p$2?.({ ReactiveElement: y$1 }), (a$1.reactiveElementVersions ??= []).push("2.1.2");
//#endregion
//#region ../../node_modules/.pnpm/lit-html@3.3.3/node_modules/lit-html/lit-html.js
/**
* @license
* Copyright 2017 Google LLC
* SPDX-License-Identifier: BSD-3-Clause
*/
const t$3 = globalThis, i$3 = (t) => t, s$3 = t$3.trustedTypes, e$4 = s$3 ? s$3.createPolicy("lit-html", { createHTML: (t) => t }) : void 0, h$3 = "$lit$", o$4 = `lit$${Math.random().toFixed(9).slice(2)}$`, n$3 = "?" + o$4, r$2 = `<${n$3}>`, l$1 = document, c$2 = () => l$1.createComment(""), a = (t) => null === t || "object" != typeof t && "function" != typeof t, u$1 = Array.isArray, d$1 = (t) => u$1(t) || "function" == typeof t?.[Symbol.iterator], f$2 = "[ 	\n\f\r]", v$1 = /<(?:(!--|\/[^a-zA-Z])|(\/?[a-zA-Z][^>\s]*)|(\/?$))/g, _ = /-->/g, m$1 = />/g, p$1 = RegExp(`>|${f$2}(?:([^\\s"'>=/]+)(${f$2}*=${f$2}*(?:[^ \t\n\f\r"'\`<>=]|("|')|))|$)`, "g"), g = /'/g, $ = /"/g, y = /^(?:script|style|textarea|title)$/i, x = (t) => (i, ...s) => ({
	_$litType$: t,
	strings: i,
	values: s
}), b = x(1), w = x(2), E = Symbol.for("lit-noChange"), A = Symbol.for("lit-nothing"), C = /* @__PURE__ */ new WeakMap(), P = l$1.createTreeWalker(l$1, 129);
function V(t, i) {
	if (!u$1(t) || !t.hasOwnProperty("raw")) throw Error("invalid template strings array");
	return void 0 !== e$4 ? e$4.createHTML(i) : i;
}
const N = (t, i) => {
	const s = t.length - 1, e = [];
	let n, l = 2 === i ? "<svg>" : 3 === i ? "<math>" : "", c = v$1;
	for (let i = 0; i < s; i++) {
		const s = t[i];
		let a, u, d = -1, f = 0;
		for (; f < s.length && (c.lastIndex = f, u = c.exec(s), null !== u);) f = c.lastIndex, c === v$1 ? "!--" === u[1] ? c = _ : void 0 !== u[1] ? c = m$1 : void 0 !== u[2] ? (y.test(u[2]) && (n = RegExp("</" + u[2], "g")), c = p$1) : void 0 !== u[3] && (c = p$1) : c === p$1 ? ">" === u[0] ? (c = n ?? v$1, d = -1) : void 0 === u[1] ? d = -2 : (d = c.lastIndex - u[2].length, a = u[1], c = void 0 === u[3] ? p$1 : "\"" === u[3] ? $ : g) : c === $ || c === g ? c = p$1 : c === _ || c === m$1 ? c = v$1 : (c = p$1, n = void 0);
		const x = c === p$1 && t[i + 1].startsWith("/>") ? " " : "";
		l += c === v$1 ? s + r$2 : d >= 0 ? (e.push(a), s.slice(0, d) + h$3 + s.slice(d) + o$4 + x) : s + o$4 + (-2 === d ? i : x);
	}
	return [V(t, l + (t[s] || "<?>") + (2 === i ? "</svg>" : 3 === i ? "</math>" : "")), e];
};
var S = class S {
	constructor({ strings: t, _$litType$: i }, e) {
		let r;
		this.parts = [];
		let l = 0, a = 0;
		const u = t.length - 1, d = this.parts, [f, v] = N(t, i);
		if (this.el = S.createElement(f, e), P.currentNode = this.el.content, 2 === i || 3 === i) {
			const t = this.el.content.firstChild;
			t.replaceWith(...t.childNodes);
		}
		for (; null !== (r = P.nextNode()) && d.length < u;) {
			if (1 === r.nodeType) {
				if (r.hasAttributes()) for (const t of r.getAttributeNames()) if (t.endsWith(h$3)) {
					const i = v[a++], s = r.getAttribute(t).split(o$4), e = /([.?@])?(.*)/.exec(i);
					d.push({
						type: 1,
						index: l,
						name: e[2],
						strings: s,
						ctor: "." === e[1] ? I : "?" === e[1] ? L : "@" === e[1] ? z : H
					}), r.removeAttribute(t);
				} else t.startsWith(o$4) && (d.push({
					type: 6,
					index: l
				}), r.removeAttribute(t));
				if (y.test(r.tagName)) {
					const t = r.textContent.split(o$4), i = t.length - 1;
					if (i > 0) {
						r.textContent = s$3 ? s$3.emptyScript : "";
						for (let s = 0; s < i; s++) r.append(t[s], c$2()), P.nextNode(), d.push({
							type: 2,
							index: ++l
						});
						r.append(t[i], c$2());
					}
				}
			} else if (8 === r.nodeType) if (r.data === n$3) d.push({
				type: 2,
				index: l
			});
			else {
				let t = -1;
				for (; -1 !== (t = r.data.indexOf(o$4, t + 1));) d.push({
					type: 7,
					index: l
				}), t += o$4.length - 1;
			}
			l++;
		}
	}
	static createElement(t, i) {
		const s = l$1.createElement("template");
		return s.innerHTML = t, s;
	}
};
function M$1(t, i, s = t, e) {
	if (i === E) return i;
	let h = void 0 !== e ? s._$Co?.[e] : s._$Cl;
	const o = a(i) ? void 0 : i._$litDirective$;
	return h?.constructor !== o && (h?._$AO?.(!1), void 0 === o ? h = void 0 : (h = new o(t), h._$AT(t, s, e)), void 0 !== e ? (s._$Co ??= [])[e] = h : s._$Cl = h), void 0 !== h && (i = M$1(t, h._$AS(t, i.values), h, e)), i;
}
var R = class {
	constructor(t, i) {
		this._$AV = [], this._$AN = void 0, this._$AD = t, this._$AM = i;
	}
	get parentNode() {
		return this._$AM.parentNode;
	}
	get _$AU() {
		return this._$AM._$AU;
	}
	u(t) {
		const { el: { content: i }, parts: s } = this._$AD, e = (t?.creationScope ?? l$1).importNode(i, !0);
		P.currentNode = e;
		let h = P.nextNode(), o = 0, n = 0, r = s[0];
		for (; void 0 !== r;) {
			if (o === r.index) {
				let i;
				2 === r.type ? i = new k(h, h.nextSibling, this, t) : 1 === r.type ? i = new r.ctor(h, r.name, r.strings, this, t) : 6 === r.type && (i = new Z(h, this, t)), this._$AV.push(i), r = s[++n];
			}
			o !== r?.index && (h = P.nextNode(), o++);
		}
		return P.currentNode = l$1, e;
	}
	p(t) {
		let i = 0;
		for (const s of this._$AV) void 0 !== s && (void 0 !== s.strings ? (s._$AI(t, s, i), i += s.strings.length - 2) : s._$AI(t[i])), i++;
	}
};
var k = class k {
	get _$AU() {
		return this._$AM?._$AU ?? this._$Cv;
	}
	constructor(t, i, s, e) {
		this.type = 2, this._$AH = A, this._$AN = void 0, this._$AA = t, this._$AB = i, this._$AM = s, this.options = e, this._$Cv = e?.isConnected ?? !0;
	}
	get parentNode() {
		let t = this._$AA.parentNode;
		const i = this._$AM;
		return void 0 !== i && 11 === t?.nodeType && (t = i.parentNode), t;
	}
	get startNode() {
		return this._$AA;
	}
	get endNode() {
		return this._$AB;
	}
	_$AI(t, i = this) {
		t = M$1(this, t, i), a(t) ? t === A || null == t || "" === t ? (this._$AH !== A && this._$AR(), this._$AH = A) : t !== this._$AH && t !== E && this._(t) : void 0 !== t._$litType$ ? this.$(t) : void 0 !== t.nodeType ? this.T(t) : d$1(t) ? this.k(t) : this._(t);
	}
	O(t) {
		return this._$AA.parentNode.insertBefore(t, this._$AB);
	}
	T(t) {
		this._$AH !== t && (this._$AR(), this._$AH = this.O(t));
	}
	_(t) {
		this._$AH !== A && a(this._$AH) ? this._$AA.nextSibling.data = t : this.T(l$1.createTextNode(t)), this._$AH = t;
	}
	$(t) {
		const { values: i, _$litType$: s } = t, e = "number" == typeof s ? this._$AC(t) : (void 0 === s.el && (s.el = S.createElement(V(s.h, s.h[0]), this.options)), s);
		if (this._$AH?._$AD === e) this._$AH.p(i);
		else {
			const t = new R(e, this), s = t.u(this.options);
			t.p(i), this.T(s), this._$AH = t;
		}
	}
	_$AC(t) {
		let i = C.get(t.strings);
		return void 0 === i && C.set(t.strings, i = new S(t)), i;
	}
	k(t) {
		u$1(this._$AH) || (this._$AH = [], this._$AR());
		const i = this._$AH;
		let s, e = 0;
		for (const h of t) e === i.length ? i.push(s = new k(this.O(c$2()), this.O(c$2()), this, this.options)) : s = i[e], s._$AI(h), e++;
		e < i.length && (this._$AR(s && s._$AB.nextSibling, e), i.length = e);
	}
	_$AR(t = this._$AA.nextSibling, s) {
		for (this._$AP?.(!1, !0, s); t !== this._$AB;) {
			const s = i$3(t).nextSibling;
			i$3(t).remove(), t = s;
		}
	}
	setConnected(t) {
		void 0 === this._$AM && (this._$Cv = t, this._$AP?.(t));
	}
};
var H = class {
	get tagName() {
		return this.element.tagName;
	}
	get _$AU() {
		return this._$AM._$AU;
	}
	constructor(t, i, s, e, h) {
		this.type = 1, this._$AH = A, this._$AN = void 0, this.element = t, this.name = i, this._$AM = e, this.options = h, s.length > 2 || "" !== s[0] || "" !== s[1] ? (this._$AH = Array(s.length - 1).fill(/* @__PURE__ */ new String()), this.strings = s) : this._$AH = A;
	}
	_$AI(t, i = this, s, e) {
		const h = this.strings;
		let o = !1;
		if (void 0 === h) t = M$1(this, t, i, 0), o = !a(t) || t !== this._$AH && t !== E, o && (this._$AH = t);
		else {
			const e = t;
			let n, r;
			for (t = h[0], n = 0; n < h.length - 1; n++) r = M$1(this, e[s + n], i, n), r === E && (r = this._$AH[n]), o ||= !a(r) || r !== this._$AH[n], r === A ? t = A : t !== A && (t += (r ?? "") + h[n + 1]), this._$AH[n] = r;
		}
		o && !e && this.j(t);
	}
	j(t) {
		t === A ? this.element.removeAttribute(this.name) : this.element.setAttribute(this.name, t ?? "");
	}
};
var I = class extends H {
	constructor() {
		super(...arguments), this.type = 3;
	}
	j(t) {
		this.element[this.name] = t === A ? void 0 : t;
	}
};
var L = class extends H {
	constructor() {
		super(...arguments), this.type = 4;
	}
	j(t) {
		this.element.toggleAttribute(this.name, !!t && t !== A);
	}
};
var z = class extends H {
	constructor(t, i, s, e, h) {
		super(t, i, s, e, h), this.type = 5;
	}
	_$AI(t, i = this) {
		if ((t = M$1(this, t, i, 0) ?? A) === E) return;
		const s = this._$AH, e = t === A && s !== A || t.capture !== s.capture || t.once !== s.once || t.passive !== s.passive, h = t !== A && (s === A || e);
		e && this.element.removeEventListener(this.name, this, s), h && this.element.addEventListener(this.name, this, t), this._$AH = t;
	}
	handleEvent(t) {
		"function" == typeof this._$AH ? this._$AH.call(this.options?.host ?? this.element, t) : this._$AH.handleEvent(t);
	}
};
var Z = class {
	constructor(t, i, s) {
		this.element = t, this.type = 6, this._$AN = void 0, this._$AM = i, this.options = s;
	}
	get _$AU() {
		return this._$AM._$AU;
	}
	_$AI(t) {
		M$1(this, t);
	}
};
const j$1 = {
	M: h$3,
	P: o$4,
	A: n$3,
	C: 1,
	L: N,
	R,
	D: d$1,
	V: M$1,
	I: k,
	H,
	N: L,
	U: z,
	B: I,
	F: Z
}, B = t$3.litHtmlPolyfillSupport;
B?.(S, k), (t$3.litHtmlVersions ??= []).push("3.3.3");
const D = (t, i, s) => {
	const e = s?.renderBefore ?? i;
	let h = e._$litPart$;
	if (void 0 === h) {
		const t = s?.renderBefore ?? null;
		e._$litPart$ = h = new k(i.insertBefore(c$2(), t), t, void 0, s ?? {});
	}
	return h._$AI(t), h;
};
//#endregion
//#region ../../node_modules/.pnpm/lit-element@4.2.2/node_modules/lit-element/lit-element.js
/**
* @license
* Copyright 2017 Google LLC
* SPDX-License-Identifier: BSD-3-Clause
*/ const s$2 = globalThis;
var i$2 = class extends y$1 {
	constructor() {
		super(...arguments), this.renderOptions = { host: this }, this._$Do = void 0;
	}
	createRenderRoot() {
		const t = super.createRenderRoot();
		return this.renderOptions.renderBefore ??= t.firstChild, t;
	}
	update(t) {
		const r = this.render();
		this.hasUpdated || (this.renderOptions.isConnected = this.isConnected), super.update(t), this._$Do = D(r, this.renderRoot, this.renderOptions);
	}
	connectedCallback() {
		super.connectedCallback(), this._$Do?.setConnected(!0);
	}
	disconnectedCallback() {
		super.disconnectedCallback(), this._$Do?.setConnected(!1);
	}
	render() {
		return E;
	}
};
i$2._$litElement$ = !0, i$2["finalized"] = !0, s$2.litElementHydrateSupport?.({ LitElement: i$2 });
const o$3 = s$2.litElementPolyfillSupport;
o$3?.({ LitElement: i$2 });
(s$2.litElementVersions ??= []).push("4.2.2");
//#endregion
//#region ../schema/dist/index.js
const DATA_READ_RPC_ALLOWLIST = [
	"health",
	"system-presence",
	"usage.status",
	"usage.cost",
	"agents.list",
	"sessions.list",
	"sessions.resolve",
	"sessions.get",
	"sessions.usage",
	"sessions.usage.timeseries",
	"sessions.usage.logs",
	"node.list",
	"node.describe",
	"cron.get",
	"cron.list",
	"cron.status",
	"cron.runs",
	"dashboard.connector.list"
];
const STREAM_EVENT_ALLOWLIST$1 = [
	"presence",
	"sessions.changed",
	"boardstate.changed"
];
const COMPUTED_OPS = [
	"sum",
	"avg",
	"min",
	"max",
	"last",
	"count",
	"pick",
	"format"
];
var DashboardBindingResolutionError = class extends Error {
	code;
	constructor(code, message) {
		super(message);
		this.code = code;
		this.name = "DashboardBindingResolutionError";
	}
};
function hasControlCharacter(value) {
	for (const char of value) {
		const code = char.charCodeAt(0);
		if (code < 32 || code === 127) return true;
	}
	return false;
}
function normalizeDashboardDataLogicalPath(value) {
	if (value.startsWith("/") || /^([a-zA-Z]:[\\/]|[\\/])/.test(value) || hasControlCharacter(value)) throw new DashboardBindingResolutionError("binding_invalid", "file binding path is invalid");
	const parts = value.replaceAll("\\", "/").split("/").filter(Boolean);
	if (parts.length === 0 || parts.some((part) => part === "." || part === ".." || part.includes(":"))) throw new DashboardBindingResolutionError("binding_invalid", "file binding path is invalid");
	return parts.join("/");
}
const TAB_SLUG_PATTERN = /^[a-z0-9-]{1,40}$/;
const ACTOR_PATTERN = /^(user|system|agent:[A-Za-z0-9._-]{1,64})$/;
const AGENT_ACTOR_PATTERN = /^agent:[A-Za-z0-9._-]{1,64}$/;
const TAB_VISIBILITY_VALUES = /* @__PURE__ */ new Set(["shared", "private"]);
/** Bounded opaque operator-identity string (e.g. `device:<id>`). */
const TAB_OWNER_PATTERN = /^[A-Za-z0-9:._-]{1,128}$/;
const WIDGET_ID_PATTERN = /^[A-Za-z0-9_-]{1,48}$/;
const BUILTIN_KIND_PATTERN = /^builtin:(stat-card|markdown|table|iframe-embed|sessions|usage|cron|instances|activity|chart|notes|action-form|action-button|preview|agent-status|approvals|chat)$/;
const CUSTOM_KIND_PATTERN = /^custom:[A-Za-z0-9._-]{1,64}$/;
const CUSTOM_WIDGET_NAME_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;
const CONNECTOR_NAME_PATTERN$1 = /^[A-Za-z0-9._-]{1,64}$/;
const CONNECTOR_TOOL_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;
const GRANT_TOOL_ID_PATTERN$1 = /^[A-Za-z0-9._-]{1,64}:[A-Za-z0-9._-]{1,64}$/;
const GRANT_TOOL_ID_MAX_LENGTH$1 = 64;
const TOOLS_HASH_PATTERN = /^[A-Za-z0-9._+/=-]{1,128}$/;
const MAX_ARGS_BINDING_BYTES = 8 * 1024;
const BINDING_ID_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;
const MAX_STATIC_BINDING_BYTES = 8 * 1024;
const MAX_COMPUTED_INPUTS = 32;
const ISO_TIMESTAMP_PATTERN = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$/;
const ACTION_FORM_FIELD_NAME_PATTERN = /^[A-Za-z0-9_]{1,32}$/;
const ACTION_FORM_SLOT_PATTERN = /\{([A-Za-z0-9_]+)\}/g;
const ACTION_FORM_MAX_TEMPLATE_CHARS = 2e3;
const ACTION_FORM_MAX_FIELDS = 8;
const ACTION_FORM_MAX_OPTIONS = 20;
const ACTION_FORM_MAX_FIELD_MAX_LENGTH = 1e3;
const ACTION_FORM_FIELD_TYPES = [
	"text",
	"number",
	"select"
];
function isRecord$1$5(value) {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
function assertRecord$1(value, path) {
	if (!isRecord$1$5(value)) throw new Error(`${path} must be an object`);
	return value;
}
function assertKnownKeys$1(record, allowed, path) {
	for (const key of Object.keys(record)) if (!allowed.includes(key)) throw new Error(`${path}.${key} is not allowed`);
}
function requireString$1(record, key, path) {
	const value = record[key];
	if (typeof value !== "string") throw new Error(`${path}.${key} must be a string`);
	return value;
}
function optionalString$1(record, key, path) {
	const value = record[key];
	if (value === void 0) return;
	if (typeof value !== "string") throw new Error(`${path}.${key} must be a string`);
	return value;
}
function requireBoolean(record, key, path) {
	const value = record[key];
	if (typeof value !== "boolean") throw new Error(`${path}.${key} must be a boolean`);
	return value;
}
function requireArray(value, path) {
	if (!Array.isArray(value)) throw new Error(`${path} must be an array`);
	return value;
}
function validateActor(value, path) {
	if (typeof value !== "string" || !ACTOR_PATTERN.test(value)) throw new Error(`${path} createdBy is invalid`);
	return value;
}
function assertIntegerRange(value, path, min, max) {
	if (!Number.isInteger(value) || value < min || value > max) throw new Error(`${path} must be an integer from ${min} to ${max}`);
	return value;
}
function validateGrid(value, path) {
	const record = assertRecord$1(value, path);
	assertKnownKeys$1(record, [
		"x",
		"y",
		"w",
		"h"
	], path);
	const grid = {
		x: assertIntegerRange(record.x, `${path}.x`, 0, 11),
		y: assertIntegerRange(record.y, `${path}.y`, 0, 499),
		w: assertIntegerRange(record.w, `${path}.w`, 1, 12),
		h: assertIntegerRange(record.h, `${path}.h`, 1, 20)
	};
	if (grid.x + grid.w > 12) throw new Error(`${path}.x + w must be 12 or less`);
	return grid;
}
function assertJsonValue(value, path) {
	if (value === null || typeof value === "string" || typeof value === "boolean" || typeof value === "number" && Number.isFinite(value)) return value;
	if (Array.isArray(value)) return value.map((entry, index) => assertJsonValue(entry, `${path}[${index}]`));
	if (isRecord$1$5(value)) {
		const next = {};
		for (const [key, entry] of Object.entries(value)) next[key] = assertJsonValue(entry, `${path}.${key}`);
		return next;
	}
	throw new Error(`${path} must be JSON-serializable`);
}
function serializedBytes(value) {
	return new TextEncoder().encode(JSON.stringify(value)).length;
}
function validateBinding(value, path) {
	const record = assertRecord$1(value, path);
	const source = requireString$1(record, "source", path);
	if (source === "rpc") {
		assertKnownKeys$1(record, ["source", "method"], path);
		const method = requireString$1(record, "method", path);
		if (!DATA_READ_RPC_ALLOWLIST.includes(method)) throw new Error(`${path}.method is not allowlisted`);
		return {
			source,
			method
		};
	}
	if (source === "file") {
		assertKnownKeys$1(record, [
			"source",
			"path",
			"pointer"
		], path);
		const bindingPath = requireString$1(record, "path", path);
		normalizeDashboardDataLogicalPath(bindingPath);
		const pointer = optionalString$1(record, "pointer", path);
		return {
			source,
			path: bindingPath,
			...pointer !== void 0 ? { pointer } : {}
		};
	}
	if (source === "static") {
		assertKnownKeys$1(record, ["source", "value"], path);
		const jsonValue = assertJsonValue(record.value, `${path}.value`);
		if (serializedBytes(jsonValue) > MAX_STATIC_BINDING_BYTES) throw new Error(`${path}.value must serialize to 8 KB or less`);
		return {
			source,
			value: jsonValue
		};
	}
	if (source === "stream") {
		assertKnownKeys$1(record, [
			"source",
			"event",
			"pointer"
		], path);
		const event = requireString$1(record, "event", path);
		if (!STREAM_EVENT_ALLOWLIST$1.includes(event)) throw new Error(`${path}.event is not allowlisted`);
		const pointer = optionalString$1(record, "pointer", path);
		if (pointer !== void 0 && !pointer.startsWith("/")) throw new Error(`${path}.pointer must be a JSON pointer`);
		return {
			source,
			event,
			...pointer !== void 0 ? { pointer } : {}
		};
	}
	if (source === "computed") {
		assertKnownKeys$1(record, [
			"source",
			"op",
			"inputs",
			"arg"
		], path);
		const op = requireString$1(record, "op", path);
		if (!COMPUTED_OPS.includes(op)) throw new Error(`${path}.op is not a valid computed op`);
		const rawInputs = requireArray(record.inputs, `${path}.inputs`);
		if (rawInputs.length < 1 || rawInputs.length > MAX_COMPUTED_INPUTS) throw new Error(`${path}.inputs must contain 1 to ${MAX_COMPUTED_INPUTS} entries`);
		const inputs = rawInputs.map((entry, index) => {
			if (typeof entry !== "string" || !BINDING_ID_PATTERN.test(entry)) throw new Error(`${path}.inputs[${index}] is invalid`);
			return entry;
		});
		const needsArg = op === "pick" || op === "format";
		const arg = optionalString$1(record, "arg", path);
		if (needsArg && (arg === void 0 || arg.length === 0)) throw new Error(`${path}.arg is required for the ${op} op`);
		if (!needsArg && arg !== void 0) throw new Error(`${path}.arg is not allowed for the ${op} op`);
		if (op === "pick" && arg !== void 0 && !arg.startsWith("/")) throw new Error(`${path}.arg must be a JSON pointer for the pick op`);
		return {
			source,
			op,
			inputs,
			...arg !== void 0 ? { arg } : {}
		};
	}
	if (source === "mcp") {
		assertKnownKeys$1(record, [
			"source",
			"connector",
			"tool",
			"args"
		], path);
		const connector = requireString$1(record, "connector", path);
		if (!CONNECTOR_NAME_PATTERN$1.test(connector)) throw new Error(`${path}.connector is invalid`);
		const tool = requireString$1(record, "tool", path);
		if (!CONNECTOR_TOOL_PATTERN.test(tool)) throw new Error(`${path}.tool is invalid`);
		const args = validateArgsObject(record.args, `${path}.args`);
		return {
			source,
			connector,
			tool,
			...args !== void 0 ? { args } : {}
		};
	}
	throw new Error(`${path}.source is invalid`);
}
/**
* Validate an optional `args` object (mcp binding, action-button, pending action):
* a JSON OBJECT (never a scalar/array) bounded to the 8 KB static-binding envelope.
* Returns the frozen JSON value, or `undefined` when the key is absent.
*/
function validateArgsObject(value, path) {
	if (value === void 0) return;
	const json = assertJsonValue(value, path);
	if (!isRecord$1$5(json)) throw new Error(`${path} must be an object`);
	if (serializedBytes(json) > MAX_ARGS_BINDING_BYTES) throw new Error(`${path} must serialize to 8 KB or less`);
	return json;
}
function validateBindingRecord(value, path) {
	const record = assertRecord$1(value, path);
	const bindings = {};
	for (const [key, entry] of Object.entries(record)) {
		if (!BINDING_ID_PATTERN.test(key)) throw new Error(`${path}.${key} binding id is invalid`);
		bindings[key] = validateBinding(entry, `${path}.${key}`);
	}
	for (const [key, binding] of Object.entries(bindings)) {
		if (binding.source !== "computed") continue;
		for (const input of binding.inputs) {
			const target = bindings[input];
			if (!target) throw new Error(`${path}.${key}.inputs references unknown binding: ${input}`);
			if (target.source === "computed") throw new Error(`${path}.${key}.inputs may not reference another computed binding: ${input}`);
		}
	}
	return bindings;
}
function validateEphemeral(value, path) {
	const record = assertRecord$1(value, path);
	assertKnownKeys$1(record, ["expiresAt"], path);
	const expiresAt = requireString$1(record, "expiresAt", path);
	if (!ISO_TIMESTAMP_PATTERN.test(expiresAt) || Number.isNaN(Date.parse(expiresAt))) throw new Error(`${path}.expiresAt must be an ISO 8601 timestamp`);
	return { expiresAt };
}
/**
* Write-time validation for a `builtin:action-form` widget's props. The template
* is authored here (not at click time); each `{slot}` MUST name a declared field,
* so an operator-approved form can never interpolate an undeclared value. Field
* values are supplied at click time and are separately typed/length-capped by the
* renderer — this gate only bounds the authored template + field set.
*/
function validateActionFormProps(value, path) {
	const record = assertRecord$1(value, path);
	assertKnownKeys$1(record, [
		"template",
		"fields",
		"buttonLabel",
		"mode",
		"connector",
		"tool",
		"argsFrom"
	], path);
	const template = requireString$1(record, "template", path);
	if (template.length < 1 || template.length > ACTION_FORM_MAX_TEMPLATE_CHARS) throw new Error(`${path}.template must be 1-${ACTION_FORM_MAX_TEMPLATE_CHARS} characters`);
	const fields = requireArray(record.fields, `${path}.fields`);
	if (fields.length < 1 || fields.length > ACTION_FORM_MAX_FIELDS) throw new Error(`${path}.fields must contain 1 to ${ACTION_FORM_MAX_FIELDS} entries`);
	const names = /* @__PURE__ */ new Set();
	fields.forEach((field, index) => {
		const fieldPath = `${path}.fields[${index}]`;
		const fieldRecord = assertRecord$1(field, fieldPath);
		assertKnownKeys$1(fieldRecord, [
			"name",
			"label",
			"type",
			"options",
			"maxLength"
		], fieldPath);
		const name = requireString$1(fieldRecord, "name", fieldPath);
		if (!ACTION_FORM_FIELD_NAME_PATTERN.test(name)) throw new Error(`${fieldPath}.name is invalid`);
		if (names.has(name)) throw new Error(`${fieldPath}.name is a duplicate: ${name}`);
		names.add(name);
		const label = requireString$1(fieldRecord, "label", fieldPath);
		if (label.length < 1 || label.length > 80) throw new Error(`${fieldPath}.label must be 1-80 characters`);
		const type = requireString$1(fieldRecord, "type", fieldPath);
		if (!ACTION_FORM_FIELD_TYPES.includes(type)) throw new Error(`${fieldPath}.type must be text, number, or select`);
		if (type === "select") {
			const options = requireArray(fieldRecord.options, `${fieldPath}.options`);
			if (options.length < 1 || options.length > ACTION_FORM_MAX_OPTIONS) throw new Error(`${fieldPath}.options must contain 1 to ${ACTION_FORM_MAX_OPTIONS} entries`);
			options.forEach((option, optionIndex) => {
				if (typeof option !== "string" || option.length < 1 || option.length > 80) throw new Error(`${fieldPath}.options[${optionIndex}] must be a 1-80 character string`);
			});
		} else if (fieldRecord.options !== void 0) throw new Error(`${fieldPath}.options is only allowed for select fields`);
		if (fieldRecord.maxLength !== void 0) assertIntegerRange(fieldRecord.maxLength, `${fieldPath}.maxLength`, 1, ACTION_FORM_MAX_FIELD_MAX_LENGTH);
	});
	if (record.buttonLabel !== void 0) {
		const buttonLabel = requireString$1(record, "buttonLabel", path);
		if (buttonLabel.length < 1 || buttonLabel.length > 40) throw new Error(`${path}.buttonLabel must be 1-40 characters`);
	}
	for (const match of template.matchAll(ACTION_FORM_SLOT_PATTERN)) {
		const slot = match[1];
		if (!names.has(slot)) throw new Error(`${path}.template references unknown field: {${slot}}`);
	}
	const mode = optionalString$1(record, "mode", path);
	if (mode !== void 0 && mode !== "prompt" && mode !== "tool") throw new Error(`${path}.mode must be "prompt" or "tool"`);
	if (mode === "tool") {
		const connector = requireString$1(record, "connector", path);
		if (!CONNECTOR_NAME_PATTERN$1.test(connector)) throw new Error(`${path}.connector is invalid`);
		const tool = requireString$1(record, "tool", path);
		if (!CONNECTOR_TOOL_PATTERN.test(tool)) throw new Error(`${path}.tool is invalid`);
		if (record.argsFrom !== void 0) {
			const argsFrom = assertRecord$1(record.argsFrom, `${path}.argsFrom`);
			const mappings = Object.entries(argsFrom);
			if (mappings.length > ACTION_FORM_MAX_FIELDS) throw new Error(`${path}.argsFrom must contain at most ${ACTION_FORM_MAX_FIELDS} entries`);
			for (const [argName, fieldName] of mappings) {
				if (!ACTION_FORM_FIELD_NAME_PATTERN.test(argName)) throw new Error(`${path}.argsFrom key is invalid: ${argName}`);
				if (typeof fieldName !== "string" || !names.has(fieldName)) throw new Error(`${path}.argsFrom references unknown field: ${String(fieldName)}`);
			}
		}
	} else for (const key of [
		"connector",
		"tool",
		"argsFrom"
	]) if (record[key] !== void 0) throw new Error(`${path}.${key} is only allowed when mode is "tool"`);
}
/**
* Write-time validation for a `builtin:action-button` widget's props (SPEC §17 v2):
* a one-click invocation of a granted external tool with fixed `args`. Shape-only —
* the actual (server-gated) invocation + pending-action parking land with #44/#41.
*/
function validateActionButtonProps(value, path) {
	const record = assertRecord$1(value, path);
	assertKnownKeys$1(record, [
		"connector",
		"tool",
		"args",
		"label"
	], path);
	const connector = requireString$1(record, "connector", path);
	if (!CONNECTOR_NAME_PATTERN$1.test(connector)) throw new Error(`${path}.connector is invalid`);
	const tool = requireString$1(record, "tool", path);
	if (!CONNECTOR_TOOL_PATTERN.test(tool)) throw new Error(`${path}.tool is invalid`);
	validateArgsObject(record.args, `${path}.args`);
	const label = optionalString$1(record, "label", path);
	if (label !== void 0 && (label.length < 1 || label.length > 40)) throw new Error(`${path}.label must be 1-40 characters`);
}
function validateWidget(value, path) {
	const record = assertRecord$1(value, path);
	assertKnownKeys$1(record, [
		"id",
		"kind",
		"title",
		"grid",
		"collapsed",
		"hidden",
		"bindings",
		"props",
		"ephemeral"
	], path);
	const id = requireString$1(record, "id", path);
	if (!WIDGET_ID_PATTERN.test(id)) throw new Error(`${path}.id is invalid`);
	const kind = requireString$1(record, "kind", path);
	if (!BUILTIN_KIND_PATTERN.test(kind) && !CUSTOM_KIND_PATTERN.test(kind)) throw new Error(`${path}.kind is invalid`);
	const title = optionalString$1(record, "title", path);
	if (title !== void 0 && title.length > 80) throw new Error(`${path}.title must be 80 characters or fewer`);
	const bindings = record.bindings === void 0 ? void 0 : validateBindingRecord(record.bindings, `${path}.bindings`);
	const props = record.props === void 0 ? void 0 : assertJsonValue(record.props, `${path}.props`);
	const ephemeral = record.ephemeral === void 0 ? void 0 : validateEphemeral(record.ephemeral, `${path}.ephemeral`);
	if (kind === "builtin:action-form") validateActionFormProps(props, `${path}.props`);
	if (kind === "builtin:action-button") validateActionButtonProps(props, `${path}.props`);
	return {
		id,
		kind,
		...title !== void 0 ? { title } : {},
		grid: validateGrid(record.grid, `${path}.grid`),
		collapsed: requireBoolean(record, "collapsed", path),
		hidden: requireBoolean(record, "hidden", path),
		...bindings !== void 0 ? { bindings } : {},
		...props !== void 0 ? { props } : {},
		...ephemeral !== void 0 ? { ephemeral } : {}
	};
}
function validateTabLayout(value, path) {
	if (value === void 0) return;
	if (value !== "grid" && value !== "full") throw new Error(`${path}.layout must be "grid" or "full"`);
	return value;
}
function validateVisibility(value, path) {
	if (value === void 0) return;
	if (typeof value !== "string" || !TAB_VISIBILITY_VALUES.has(value)) throw new Error(`${path}.visibility must be "shared" or "private"`);
	return value;
}
function validateTab(value, path) {
	const record = assertRecord$1(value, path);
	assertKnownKeys$1(record, [
		"slug",
		"title",
		"icon",
		"hidden",
		"layout",
		"createdBy",
		"visibility",
		"owner",
		"widgets"
	], path);
	const slug = requireString$1(record, "slug", path);
	if (!TAB_SLUG_PATTERN.test(slug)) throw new Error(`${path}.slug is invalid`);
	const title = requireString$1(record, "title", path);
	if (title.length < 1 || title.length > 80) throw new Error(`${path}.title must be 1-80 characters`);
	const icon = optionalString$1(record, "icon", path);
	if (icon !== void 0 && icon.length > 40) throw new Error(`${path}.icon must be 40 characters or fewer`);
	const layout = validateTabLayout(record.layout, path);
	const visibility = validateVisibility(record.visibility, path);
	const owner = optionalString$1(record, "owner", path);
	if (owner !== void 0 && !TAB_OWNER_PATTERN.test(owner)) throw new Error(`${path}.owner is invalid`);
	if (visibility === "private" && owner === void 0) throw new Error(`${path}.owner is required when the tab is private`);
	const widgets = requireArray(record.widgets, `${path}.widgets`);
	if (widgets.length > 24) throw new Error(`${path}.widgets must contain at most 24 entries`);
	return {
		slug,
		title,
		...icon !== void 0 ? { icon } : {},
		hidden: requireBoolean(record, "hidden", path),
		...layout !== void 0 ? { layout } : {},
		createdBy: validateActor(record.createdBy, `${path}.createdBy`),
		...visibility === "private" ? { visibility } : {},
		...owner !== void 0 ? { owner } : {},
		widgets: widgets.map((widget, index) => validateWidget(widget, `${path}.widgets[${index}]`))
	};
}
function validateRegistryEntry(value, path) {
	const record = assertRecord$1(value, path);
	assertKnownKeys$1(record, [
		"status",
		"createdBy",
		"approvedBy",
		"approvedAt"
	], path);
	const status = requireString$1(record, "status", path);
	if (status !== "pending" && status !== "approved" && status !== "rejected") throw new Error(`${path}.status is invalid`);
	const approvedBy = record.approvedBy === void 0 ? void 0 : validateActor(record.approvedBy, `${path}.approvedBy`);
	const approvedAt = optionalString$1(record, "approvedAt", path);
	return {
		status,
		createdBy: validateActor(record.createdBy, `${path}.createdBy`),
		...approvedBy !== void 0 ? { approvedBy } : {},
		...approvedAt !== void 0 ? { approvedAt } : {}
	};
}
function validateWidgetsRegistry(value) {
	const record = assertRecord$1(value, "widgetsRegistry");
	const registry = {};
	for (const [name, entry] of Object.entries(record)) {
		if (!CUSTOM_WIDGET_NAME_PATTERN.test(name)) throw new Error(`widgetsRegistry.${name} name is invalid`);
		registry[name] = validateRegistryEntry(entry, `widgetsRegistry.${name}`);
	}
	return registry;
}
const CAPABILITY_STATUSES$1 = /* @__PURE__ */ new Set([
	"requested",
	"granted",
	"revoked"
]);
function validateCapabilityGrant(value, path) {
	const record = assertRecord$1(value, path);
	assertKnownKeys$1(record, [
		"status",
		"methods",
		"streams",
		"tools",
		"toolsHash",
		"autoConfirm",
		"expiresAt",
		"agents",
		"description",
		"grantedBy",
		"grantedAt"
	], path);
	const status = record.status;
	if (typeof status !== "string" || !CAPABILITY_STATUSES$1.has(status)) throw new Error(`${path}.status must be requested, granted, or revoked`);
	const methods = allowlistArray(record.methods, `${path}.methods`, DATA_READ_RPC_ALLOWLIST, "allowlisted read method");
	const streams = allowlistArray(record.streams, `${path}.streams`, STREAM_EVENT_ALLOWLIST$1, "allowlisted stream channel");
	const tools = record.tools === void 0 ? void 0 : requireArray(record.tools, `${path}.tools`).map((tool, index) => {
		if (typeof tool !== "string" || tool.length > GRANT_TOOL_ID_MAX_LENGTH$1 || !GRANT_TOOL_ID_PATTERN$1.test(tool)) throw new Error(`${path}.tools[${index}] is not a valid connector:tool id`);
		return tool;
	});
	if (tools !== void 0 && new Set(tools).size !== tools.length) throw new Error(`${path}.tools contains duplicate tool ids`);
	const toolsHash = optionalString$1(record, "toolsHash", path);
	if (toolsHash !== void 0 && !TOOLS_HASH_PATTERN.test(toolsHash)) throw new Error(`${path}.toolsHash is invalid`);
	const autoConfirm = record.autoConfirm === void 0 ? void 0 : requireArray(record.autoConfirm, `${path}.autoConfirm`).map((entry, index) => {
		if (typeof entry !== "string" || entry.length > GRANT_TOOL_ID_MAX_LENGTH$1 || !GRANT_TOOL_ID_PATTERN$1.test(entry)) throw new Error(`${path}.autoConfirm[${index}] is not a valid connector:tool id`);
		return entry;
	});
	if (autoConfirm !== void 0) {
		if (new Set(autoConfirm).size !== autoConfirm.length) throw new Error(`${path}.autoConfirm contains duplicate tool ids`);
		const granted = new Set(tools ?? []);
		for (const id of autoConfirm) if (!granted.has(id)) throw new Error(`${path}.autoConfirm[${id}] is not one of the grant's tools`);
	}
	const expiresAt = optionalString$1(record, "expiresAt", path);
	if (expiresAt !== void 0 && (!ISO_TIMESTAMP_PATTERN.test(expiresAt) || Number.isNaN(Date.parse(expiresAt)))) throw new Error(`${path}.expiresAt must be an ISO 8601 timestamp`);
	const agents = record.agents === void 0 ? void 0 : requireArray(record.agents, `${path}.agents`).map((entry, index) => {
		if (typeof entry !== "string" || !AGENT_ACTOR_PATTERN.test(entry)) throw new Error(`${path}.agents[${index}] is not a valid agent actor`);
		return entry;
	});
	if (agents !== void 0) {
		if (agents.length === 0) throw new Error(`${path}.agents must be a non-empty array (omit it to allow all agents)`);
		if (new Set(agents).size !== agents.length) throw new Error(`${path}.agents contains duplicate actors`);
	}
	const description = optionalString$1(record, "description", path);
	if (description !== void 0 && description.length > 200) throw new Error(`${path}.description must be 200 characters or fewer`);
	const grantedBy = record.grantedBy === void 0 ? void 0 : validateActor(record.grantedBy, `${path}.grantedBy`);
	const grantedAt = optionalString$1(record, "grantedAt", path);
	return {
		status,
		methods,
		streams,
		...tools !== void 0 ? { tools } : {},
		...toolsHash !== void 0 ? { toolsHash } : {},
		...autoConfirm !== void 0 ? { autoConfirm } : {},
		...expiresAt !== void 0 ? { expiresAt } : {},
		...agents !== void 0 ? { agents } : {},
		...description !== void 0 ? { description } : {},
		...grantedBy !== void 0 ? { grantedBy } : {},
		...grantedAt !== void 0 ? { grantedAt } : {}
	};
}
/**
* A REQUIRED array of allowlisted string entries. Shared by a grant's
* `methods`/`streams`; an absent key rejects, exactly as pre-§17 (invariant #7 —
* verdicts on old shapes never change). Tools-only grants pass explicit [].
*/
function allowlistArray(value, path, allowlist, label) {
	return requireArray(value, path).map((entry, index) => {
		if (typeof entry !== "string" || !allowlist.includes(entry)) throw new Error(`${path}[${index}] is not an ${label}`);
		return entry;
	});
}
function validateCapabilitiesRegistry(value) {
	if (value === void 0) return {};
	const record = assertRecord$1(value, "capabilitiesRegistry");
	const registry = {};
	for (const [name, entry] of Object.entries(record)) {
		if (!CONNECTOR_NAME_PATTERN$1.test(name)) throw new Error(`capabilitiesRegistry.${name} connector name is invalid`);
		registry[name] = validateCapabilityGrant(entry, `capabilitiesRegistry.${name}`);
	}
	return registry;
}
function validatePrefs(value, tabSlugs) {
	const record = assertRecord$1(value, "prefs");
	assertKnownKeys$1(record, ["tabOrder"], "prefs");
	const tabOrder = requireArray(record.tabOrder, "prefs.tabOrder");
	const seen = /* @__PURE__ */ new Set();
	return { tabOrder: tabOrder.map((entry, index) => {
		if (typeof entry !== "string" || !TAB_SLUG_PATTERN.test(entry)) throw new Error(`prefs.tabOrder[${index}] is invalid`);
		if (!tabSlugs.has(entry)) throw new Error(`prefs.tabOrder[${index}] is not a tab slug`);
		if (seen.has(entry)) throw new Error(`prefs.tabOrder contains duplicate slug: ${entry}`);
		seen.add(entry);
		return entry;
	}) };
}
function assertUniqueTabs(tabs) {
	const slugs = /* @__PURE__ */ new Set();
	for (const tab of tabs) {
		if (slugs.has(tab.slug)) throw new Error(`duplicate tab slug: ${tab.slug}`);
		slugs.add(tab.slug);
	}
	return slugs;
}
function assertUniqueWidgets(tabs) {
	const ids = /* @__PURE__ */ new Set();
	for (const tab of tabs) for (const widget of tab.widgets) {
		if (ids.has(widget.id)) throw new Error(`duplicate widget id: ${widget.id}`);
		ids.add(widget.id);
	}
}
function validateWorkspaceDoc(value) {
	const record = assertRecord$1(value, "workspace");
	assertKnownKeys$1(record, [
		"schemaVersion",
		"workspaceVersion",
		"tabs",
		"widgetsRegistry",
		"capabilitiesRegistry",
		"prefs"
	], "workspace");
	if (record.schemaVersion !== 1) throw new Error(`schemaVersion must be 1`);
	const workspaceVersion = assertIntegerRange(record.workspaceVersion, "workspaceVersion", 0, Number.MAX_SAFE_INTEGER);
	const rawTabs = requireArray(record.tabs, "tabs");
	if (rawTabs.length > 32) throw new Error("tabs must contain at most 32 entries");
	const tabs = rawTabs.map((tab, index) => validateTab(tab, `tabs[${index}]`));
	const tabSlugs = assertUniqueTabs(tabs);
	assertUniqueWidgets(tabs);
	return {
		schemaVersion: 1,
		workspaceVersion,
		tabs,
		widgetsRegistry: validateWidgetsRegistry(record.widgetsRegistry),
		capabilitiesRegistry: validateCapabilitiesRegistry(record.capabilitiesRegistry),
		prefs: validatePrefs(record.prefs, tabSlugs)
	};
}
const RECIPE_NAME_PATTERN$1 = /^[A-Za-z0-9._-]{1,64}$/;
const CONNECTOR_NAME_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;
const GRANT_TOOL_ID_PATTERN = /^[A-Za-z0-9._-]{1,64}:[A-Za-z0-9._-]{1,64}$/;
const GRANT_TOOL_ID_MAX_LENGTH = 64;
const RECIPE_TITLE_MAX = 80;
const RECIPE_DESCRIPTION_MAX = 280;
const GRANT_LABEL_MAX = 80;
const GRANT_REASON_MAX = 200;
const MAX_CONNECTORS = 16;
const MAX_TOOLS_PER_CONNECTOR = 32;
function isRecord$6(value) {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
function assertRecord(value, path) {
	if (!isRecord$6(value)) throw new Error(`${path} must be an object`);
	return value;
}
function assertKnownKeys(record, allowed, path) {
	for (const key of Object.keys(record)) if (!allowed.includes(key)) throw new Error(`${path}.${key} is not allowed`);
}
function requireString(record, key, path) {
	const value = record[key];
	if (typeof value !== "string") throw new Error(`${path}.${key} must be a string`);
	return value;
}
function optionalString(record, key, path) {
	const value = record[key];
	if (value === void 0) return;
	if (typeof value !== "string") throw new Error(`${path}.${key} must be a string`);
	return value;
}
function optionalStringArray(record, key, path) {
	const value = record[key];
	if (value === void 0) return;
	if (!Array.isArray(value)) throw new Error(`${path}.${key} must be an array`);
	return value.map((entry, index) => {
		if (typeof entry !== "string" || entry.length === 0) throw new Error(`${path}.${key}[${index}] must be a non-empty string`);
		return entry;
	});
}
function validateGrantTool(value, connector, path) {
	const record = assertRecord(value, path);
	assertKnownKeys(record, [
		"id",
		"label",
		"readOnly"
	], path);
	const id = requireString(record, "id", path);
	if (id.length > GRANT_TOOL_ID_MAX_LENGTH || !GRANT_TOOL_ID_PATTERN.test(id)) throw new Error(`${path}.id is not a valid connector:tool id`);
	if (id.slice(0, id.indexOf(":")) !== connector) throw new Error(`${path}.id "${id}" must be namespaced under connector "${connector}"`);
	const label = requireString(record, "label", path);
	if (label.length < 1 || label.length > GRANT_LABEL_MAX) throw new Error(`${path}.label must be 1-${GRANT_LABEL_MAX} characters`);
	const readOnly = record.readOnly;
	if (readOnly !== void 0 && typeof readOnly !== "boolean") throw new Error(`${path}.readOnly must be a boolean`);
	return {
		id,
		label,
		...readOnly !== void 0 ? { readOnly } : {}
	};
}
function validateConnectorGrant(value, connector, path) {
	const record = assertRecord(value, path);
	assertKnownKeys(record, [
		"label",
		"reason",
		"methods",
		"streams",
		"tools"
	], path);
	const label = requireString(record, "label", path);
	if (label.length < 1 || label.length > GRANT_LABEL_MAX) throw new Error(`${path}.label must be 1-${GRANT_LABEL_MAX} characters`);
	const reason = optionalString(record, "reason", path);
	if (reason !== void 0 && reason.length > GRANT_REASON_MAX) throw new Error(`${path}.reason must be ${GRANT_REASON_MAX} characters or fewer`);
	const methods = optionalStringArray(record, "methods", path);
	const streams = optionalStringArray(record, "streams", path);
	let tools;
	if (record.tools !== void 0) {
		if (!Array.isArray(record.tools)) throw new Error(`${path}.tools must be an array`);
		if (record.tools.length > MAX_TOOLS_PER_CONNECTOR) throw new Error(`${path}.tools must contain at most ${MAX_TOOLS_PER_CONNECTOR} entries`);
		tools = record.tools.map((tool, index) => validateGrantTool(tool, connector, `${path}.tools[${index}]`));
		const ids = tools.map((tool) => tool.id);
		if (new Set(ids).size !== ids.length) throw new Error(`${path}.tools contains duplicate tool ids`);
	}
	if (!((methods?.length ?? 0) > 0 || (streams?.length ?? 0) > 0 || (tools?.length ?? 0) > 0)) throw new Error(`${path} must request at least one tool, method, or stream`);
	return {
		label,
		...reason !== void 0 ? { reason } : {},
		...methods !== void 0 ? { methods } : {},
		...streams !== void 0 ? { streams } : {},
		...tools !== void 0 ? { tools } : {}
	};
}
function validateGrantsManifest(value, path) {
	if (value === void 0) return {};
	const record = assertRecord(value, path);
	if (Object.keys(record).length > MAX_CONNECTORS) throw new Error(`${path} must reference at most ${MAX_CONNECTORS} connectors`);
	const manifest = {};
	for (const [connector, entry] of Object.entries(record)) {
		if (!CONNECTOR_NAME_PATTERN.test(connector)) throw new Error(`${path}.${connector} connector name is invalid`);
		manifest[connector] = validateConnectorGrant(entry, connector, `${path}.${connector}`);
	}
	return manifest;
}
/**
* Validate a template recipe (SPEC / issue #60). Throws on any malformed field so the
* honesty gate (`pnpm build:registry` + the templates test) can trust every shipped
* recipe. The embedded `doc` runs through the SAME `validateWorkspaceDoc` as every write
* path — a recipe can never smuggle a doc the store would reject. This validates SHAPE;
* the install-time re-pend (no recipe arrives pre-granted) is enforced downstream at the
* store, not here.
*/
function validateRecipe(value) {
	const record = assertRecord(value, "recipe");
	assertKnownKeys(record, [
		"recipeVersion",
		"name",
		"title",
		"description",
		"doc",
		"grantsManifest"
	], "recipe");
	if (record.recipeVersion !== 1) throw new Error(`recipe.recipeVersion must be 1`);
	const name = requireString(record, "name", "recipe");
	if (!RECIPE_NAME_PATTERN$1.test(name)) throw new Error("recipe.name is invalid");
	const title = requireString(record, "title", "recipe");
	if (title.length < 1 || title.length > RECIPE_TITLE_MAX) throw new Error(`recipe.title must be 1-${RECIPE_TITLE_MAX} characters`);
	const description = requireString(record, "description", "recipe");
	if (description.length < 1 || description.length > RECIPE_DESCRIPTION_MAX) throw new Error(`recipe.description must be 1-${RECIPE_DESCRIPTION_MAX} characters`);
	if (record.doc === void 0) throw new Error("recipe.doc is required");
	return {
		recipeVersion: 1,
		name,
		title,
		description,
		doc: validateWorkspaceDoc(record.doc),
		grantsManifest: validateGrantsManifest(record.grantsManifest, "recipe.grantsManifest")
	};
}
/** The broadcast event name every AgentStreamEvent travels under (SPEC §14.2). */
const CHAT_EVENT = "boardstate.chat.event";
//#endregion
//#region ../core/dist/index.js
/** A small grid rect helper so the examples read uniformly. */
function grid(x, y, w, h) {
	return {
		x,
		y,
		w,
		h
	};
}
const WIDGET_CATALOG = [
	{
		kind: "builtin:stat-card",
		summary: "One number that matters — a KPI with a label.",
		bindings: [{
			key: "value",
			shape: "number | string, or a structured payload + props.metric"
		}],
		props: {
			format: "\"usd\" | \"int\" | \"percent\" | \"raw\" (how the number renders)",
			metric: "when the binding resolves an object, the field name to display",
			label: "inner label (omit if it would just repeat the title)"
		},
		example: {
			id: "mrr",
			kind: "builtin:stat-card",
			title: "MRR",
			grid: grid(0, 0, 3, 2),
			collapsed: false,
			hidden: false,
			bindings: { value: {
				source: "static",
				value: 128400
			} },
			props: {
				format: "usd",
				label: "Monthly recurring revenue"
			}
		}
	},
	{
		kind: "builtin:chart",
		summary: "Trends, comparisons, budgets — a small inline chart.",
		bindings: [{
			key: "value",
			shape: "number[] (or labeled points {label,value}[])"
		}],
		props: {
			type: "\"line\" | \"bar\" | \"area\" | \"sparkline\" | \"gauge\" (default line)",
			detail: "true adds labeled axes, gridlines, and value tooltips (line/bar/area)",
			label: "sparkline only: true shows the trailing value as an end label"
		},
		example: {
			id: "revenue-trend",
			kind: "builtin:chart",
			title: "Revenue (14d)",
			grid: grid(0, 2, 8, 5),
			collapsed: false,
			hidden: false,
			bindings: { value: {
				source: "static",
				value: [
					8,
					12,
					10,
					18,
					24,
					21,
					30,
					35,
					41,
					52
				]
			} },
			props: { type: "area" }
		},
		examples: [{
			id: "signups-spark",
			kind: "builtin:chart",
			title: "Signups",
			grid: grid(0, 7, 3, 2),
			collapsed: false,
			hidden: false,
			bindings: { value: {
				source: "static",
				value: [
					12,
					9,
					14,
					11,
					17,
					15,
					22
				]
			} },
			props: {
				type: "sparkline",
				label: true
			}
		}, {
			id: "latency-detail",
			kind: "builtin:chart",
			title: "p95 latency (ms)",
			grid: grid(0, 9, 8, 5),
			collapsed: false,
			hidden: false,
			bindings: { value: {
				source: "static",
				value: [
					180,
					220,
					190,
					240,
					210,
					260,
					230
				]
			} },
			props: {
				type: "line",
				detail: true
			}
		}]
	},
	{
		kind: "builtin:table",
		summary: "Rows and columns — a compact table (keep ~10 visible rows).",
		bindings: [{
			key: "rows",
			shape: "Array<Record<string, unknown>> — NOT `value`"
		}],
		props: {
			columns: "string[] of keys to show (defaults to the first row's keys)",
			limit: "max visible rows before a “+N more” count"
		},
		example: {
			id: "recent-runs",
			kind: "builtin:table",
			title: "Recent runs",
			grid: grid(0, 7, 8, 4),
			collapsed: false,
			hidden: false,
			bindings: { rows: {
				source: "static",
				value: [{
					agent: "finance",
					task: "Q3 rollup",
					status: "done"
				}, {
					agent: "ops",
					task: "Log sweep",
					status: "running"
				}]
			} },
			props: { columns: [
				"agent",
				"task",
				"status"
			] }
		}
	},
	{
		kind: "builtin:markdown",
		summary: "Prose, explanations, small markdown tables (sanitized).",
		bindings: [{
			key: "content",
			shape: "markdown string — NOT `value`"
		}],
		props: {
			markdown: "inline markdown source (used when there is no `content` binding)",
			text: "alias for `markdown`"
		},
		example: {
			id: "summary",
			kind: "builtin:markdown",
			title: "Summary",
			grid: grid(8, 2, 4, 5),
			collapsed: false,
			hidden: false,
			props: { markdown: "## Insights\n\n- Signal up **6.5×** across 14 days.\n- Momentum late." }
		}
	},
	{
		kind: "builtin:notes",
		summary: "Operator scratch text (persisted via widget state).",
		bindings: [],
		props: { text: "starter content" },
		example: {
			id: "scratchpad",
			kind: "builtin:notes",
			title: "Notes",
			grid: grid(8, 7, 4, 4),
			collapsed: false,
			hidden: false,
			props: { text: "Jot findings here…" }
		}
	},
	{
		kind: "builtin:activity",
		summary: "An event feed — recent things that happened.",
		bindings: [{
			key: "value",
			shape: "{ entries: [{ ts, jobName, status, summary }] }"
		}],
		props: { limit: "max entries shown" },
		example: {
			id: "agent-events",
			kind: "builtin:activity",
			title: "Agent events",
			grid: grid(0, 11, 6, 4),
			collapsed: false,
			hidden: false,
			bindings: { value: {
				source: "static",
				value: { entries: [{
					ts: 17836e8,
					jobName: "finance",
					status: "ok",
					summary: "Rollup posted"
				}] }
			} }
		}
	},
	{
		kind: "builtin:action-form",
		summary: "The chat↔dashboard loop — a form that submits through the control plane.",
		bindings: [],
		props: {
			template: "the message sent on submit; `{{fieldName}}` interpolates a field (single pass)",
			fields: "array of { name, label, type: \"text\"|\"number\"|\"select\", options?, maxLength? }",
			buttonLabel: "the submit button text (optional)",
			mode: "\"prompt\" (default: submit the template to the agent) or \"tool\" (invoke a granted external tool)",
			connector: "tool mode only: the granted connector name (SPEC §17 v2)",
			tool: "tool mode only: the tool to invoke on that connector",
			argsFrom: "tool mode only: map of tool-arg name → declared field name"
		},
		example: {
			id: "ask-agent",
			kind: "builtin:action-form",
			title: "Ask the agent",
			grid: grid(0, 0, 4, 3),
			collapsed: false,
			hidden: false,
			props: {
				template: "Summarize {{topic}} for the board.",
				fields: [{
					name: "topic",
					label: "Topic",
					type: "text"
				}],
				buttonLabel: "Ask"
			}
		},
		examples: [{
			id: "file-ticket",
			kind: "builtin:action-form",
			title: "File a ticket",
			grid: grid(0, 0, 4, 4),
			collapsed: false,
			hidden: false,
			props: {
				mode: "tool",
				connector: "linear",
				tool: "create_issue",
				template: "Create issue: {title}",
				fields: [{
					name: "title",
					label: "Title",
					type: "text",
					maxLength: 120
				}, {
					name: "priority",
					label: "Priority",
					type: "select",
					options: [
						"low",
						"med",
						"high"
					]
				}],
				argsFrom: {
					title: "title",
					priority: "priority"
				},
				buttonLabel: "Create"
			}
		}]
	},
	{
		kind: "builtin:action-button",
		summary: "One click → invoke a granted external tool with fixed args (operator-confirmed).",
		bindings: [],
		props: {
			connector: "the granted connector name (SPEC §17 v2)",
			tool: "the tool to invoke on that connector",
			args: "fixed argument object passed on click (optional)",
			label: "button text (optional)"
		},
		example: {
			id: "restart-worker",
			kind: "builtin:action-button",
			title: "Restart worker",
			grid: grid(0, 0, 3, 2),
			collapsed: false,
			hidden: false,
			props: {
				connector: "officecli",
				tool: "restart_service",
				args: { service: "worker" },
				label: "Restart"
			}
		}
	},
	{
		kind: "builtin:chat",
		summary: "Talk to the agent and watch it work (ignores bindings).",
		bindings: [],
		props: { placeholder: "empty-input hint text" },
		example: {
			id: "assistant",
			kind: "builtin:chat",
			title: "Assistant",
			grid: grid(0, 0, 6, 8),
			collapsed: false,
			hidden: false,
			props: { placeholder: "Ask me to build a view…" }
		}
	}
];
/**
* Data-source builtins: these render a fixed shape fed by an allowlisted `rpc` read
* method or a `stream` binding a host wires up (not typically hand-authored with static
* data). Listed by kind + the value shape they consume; a host that has the connector
* binds them, e.g. `{ source: "rpc", method: "usage.cost" }`.
*/
const DATA_SOURCE_WIDGET_KINDS = [
	{
		kind: "builtin:sessions",
		summary: "Who/what is running.",
		valueShape: "rows { key, label, status, hasActiveRun, updatedAt }; props.limit"
	},
	{
		kind: "builtin:agent-status",
		summary: "Agents + goals/progress.",
		valueShape: "sessions shape + goal { objective, tokensUsed, tokenBudget }"
	},
	{
		kind: "builtin:usage",
		summary: "Cost/token totals.",
		valueShape: "{ totals: { totalCost, totalTokens }, days? }"
	},
	{
		kind: "builtin:cron",
		summary: "Scheduled jobs.",
		valueShape: "{ jobs: [{ id, name, enabled, state: { nextRunAtMs, lastRunStatus } }] }"
	},
	{
		kind: "builtin:instances",
		summary: "Fleet presence.",
		valueShape: "{ presence: [{ instanceId, platform, version, lastInputSeconds }] }"
	},
	{
		kind: "builtin:approvals",
		summary: "Pending widget approvals (reads the live registry; ignores bindings).",
		valueShape: "none — reads the registry"
	},
	{
		kind: "builtin:preview",
		summary: "A live page preview.",
		valueShape: "props.url (same-origin ok; cross-origin needs host opt-in)"
	},
	{
		kind: "builtin:iframe-embed",
		summary: "An embedded live page.",
		valueShape: "props.url (same-origin ok; cross-origin needs host opt-in)"
	}
];
[...WIDGET_CATALOG.map((entry) => entry.kind), ...DATA_SOURCE_WIDGET_KINDS.map((entry) => entry.kind)];
function indexWidgets(workspace) {
	const index = /* @__PURE__ */ new Map();
	for (const tab of workspace.tabs) for (const widget of tab.widgets) index.set(widget.id, {
		widget,
		tabSlug: tab.slug
	});
	return index;
}
function indexTabs(workspace) {
	return new Map(workspace.tabs.map((tab) => [tab.slug, tab]));
}
function sameRect(a, b) {
	return a.grid.x === b.grid.x && a.grid.y === b.grid.y && a.grid.w === b.grid.w && a.grid.h === b.grid.h;
}
/**
* Compute the changelist to move from `snapshot` (a past state) to `current`. A
* widget that both moved and was retitled yields two entries. Ordering is stable:
* tab changes first, then widget added/removed/moved/retitled. The view groups the
* flat list by `actor`.
*/
function computeWorkspaceDiff(snapshot, current) {
	const entries = [];
	const snapTabs = indexTabs(snapshot);
	const currTabs = indexTabs(current);
	for (const [slug, tab] of currTabs) if (!snapTabs.has(slug)) entries.push({
		kind: "tab-added",
		actor: tab.createdBy ?? null,
		id: slug,
		label: tab.title
	});
	for (const [slug, tab] of snapTabs) if (!currTabs.has(slug)) entries.push({
		kind: "tab-removed",
		actor: tab.createdBy ?? null,
		id: slug,
		label: tab.title
	});
	else {
		const currentTab = currTabs.get(slug);
		if (currentTab.title !== tab.title) entries.push({
			kind: "tab-retitled",
			actor: currentTab.createdBy ?? tab.createdBy ?? null,
			id: slug,
			label: currentTab.title,
			detail: `${tab.title} → ${currentTab.title}`
		});
	}
	const snapWidgets = indexWidgets(snapshot);
	const currWidgets = indexWidgets(current);
	for (const [id, location] of currWidgets) if (!snapWidgets.has(id)) entries.push({
		kind: "widget-added",
		actor: location.widget.createdBy ?? null,
		id,
		label: location.widget.title || id
	});
	for (const [id, location] of snapWidgets) {
		const currentLocation = currWidgets.get(id);
		if (!currentLocation) {
			entries.push({
				kind: "widget-removed",
				actor: location.widget.createdBy ?? null,
				id,
				label: location.widget.title || id
			});
			continue;
		}
		const before = location.widget;
		const after = currentLocation.widget;
		if (location.tabSlug !== currentLocation.tabSlug || !sameRect(before, after)) entries.push({
			kind: "widget-moved",
			actor: after.createdBy ?? null,
			id,
			label: after.title || id,
			detail: location.tabSlug !== currentLocation.tabSlug ? `${location.tabSlug} → ${currentLocation.tabSlug}` : void 0
		});
		if (before.title !== after.title) entries.push({
			kind: "widget-retitled",
			actor: after.createdBy ?? null,
			id,
			label: after.title || id,
			detail: `${before.title || id} → ${after.title || id}`
		});
	}
	return entries;
}
/** Group a flat changelist by `actor`, preserving first-seen actor order. */
function groupDiffByActor(entries) {
	const groups = /* @__PURE__ */ new Map();
	for (const entry of entries) {
		const bucket = groups.get(entry.actor);
		if (bucket) bucket.push(entry);
		else groups.set(entry.actor, [entry]);
	}
	return [...groups.entries()].map(([actor, grouped]) => ({
		actor,
		entries: grouped
	}));
}
function hasWidget(workspace, widgetId) {
	return workspace.tabs.some((tab) => tab.widgets.some((widget) => widget.id === widgetId));
}
/**
* Best-effort "version a widget first appeared", recovered from loaded ring
* snapshots. Returns the earliest snapshot version that contains the widget ONLY
* when an even older snapshot lacks it (so the appearance is genuinely observed
* inside the ring window); otherwise undefined — the widget predates the ring or
* no bodies are loaded yet, and the blame line falls back to provenance only.
*/
function firstSeenVersion(widgetId, snapshots) {
	const containing = snapshots.filter((snapshot) => hasWidget(snapshot.workspace, widgetId)).map((snapshot) => snapshot.version).toSorted((a, b) => a - b);
	if (containing.length === 0) return;
	const earliest = containing[0];
	return snapshots.some((snapshot) => snapshot.version < earliest) ? earliest : void 0;
}
/** Provenance is an agent authorship when the stamp is prefixed `agent:`. */
function dashboardAgentProvenance(createdBy) {
	if (typeof createdBy !== "string") return null;
	const trimmed = createdBy.trim();
	return trimmed.startsWith("agent:") ? trimmed.slice(6) || "agent" : null;
}
function isRecord$5(value) {
	return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
function readString(value, fallback = "") {
	return typeof value === "string" ? value : fallback;
}
function readNumber$1(value, fallback = 0) {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
function normalizeRect(value) {
	const record = isRecord$5(value) ? value : {};
	const w = Math.min(12, Math.max(1, Math.trunc(readNumber$1(record.w, 4))));
	const h = Math.max(1, Math.trunc(readNumber$1(record.h, 2)));
	return {
		x: Math.min(12 - w, Math.max(0, Math.trunc(readNumber$1(record.x, 0)))),
		y: Math.max(0, Math.trunc(readNumber$1(record.y, 0))),
		w,
		h
	};
}
function normalizeBinding(value) {
	if (!isRecord$5(value)) return null;
	const source = value.source;
	if (source !== "rpc" && source !== "file" && source !== "static" && source !== "stream" && source !== "computed" && source !== "mcp") return null;
	return {
		source,
		...typeof value.method === "string" ? { method: value.method } : {},
		...typeof value.path === "string" ? { path: value.path } : {},
		...typeof value.pointer === "string" ? { pointer: value.pointer } : {},
		...isRecord$5(value.params) ? { params: value.params } : {},
		..."value" in value ? { value: value.value } : {},
		...typeof value.event === "string" ? { event: value.event } : {},
		...typeof value.op === "string" ? { op: value.op } : {},
		...Array.isArray(value.inputs) ? { inputs: value.inputs.filter((input) => typeof input === "string") } : {},
		...typeof value.arg === "string" ? { arg: value.arg } : {},
		...typeof value.connector === "string" ? { connector: value.connector } : {},
		...typeof value.tool === "string" ? { tool: value.tool } : {},
		...isRecord$5(value.args) ? { args: value.args } : {}
	};
}
function normalizeBindings(value) {
	if (!isRecord$5(value)) return;
	const bindings = {};
	for (const [key, raw] of Object.entries(value)) {
		const binding = normalizeBinding(raw);
		if (binding) bindings[key] = binding;
	}
	return Object.keys(bindings).length ? bindings : void 0;
}
function normalizeWidget(value) {
	if (!isRecord$5(value)) return null;
	const id = readString(value.id).trim();
	const kind = readString(value.kind).trim();
	if (!id || !kind) return null;
	const ephemeral = normalizeEphemeral(value.ephemeral);
	return {
		id,
		kind,
		title: readString(value.title),
		grid: normalizeRect(value.grid),
		collapsed: value.collapsed === true,
		...typeof value.createdBy === "string" ? { createdBy: value.createdBy } : {},
		...normalizeBindings(value.bindings) ? { bindings: normalizeBindings(value.bindings) } : {},
		...isRecord$5(value.props) ? { props: value.props } : {},
		...ephemeral ? { ephemeral } : {}
	};
}
/** Read the ephemeral marker if present and well-formed (`{ expiresAt: string }`). */
function normalizeEphemeral(value) {
	if (!isRecord$5(value) || typeof value.expiresAt !== "string" || !value.expiresAt.trim()) return null;
	return { expiresAt: value.expiresAt };
}
function normalizeTab(value) {
	if (!isRecord$5(value)) return null;
	const slug = readString(value.slug).trim();
	if (!slug) return null;
	const widgets = Array.isArray(value.widgets) ? value.widgets.map(normalizeWidget).filter((w) => w !== null) : [];
	return {
		slug,
		title: readString(value.title, slug),
		hidden: value.hidden === true,
		widgets,
		...value.layout === "full" || value.layout === "grid" ? { layout: value.layout } : {},
		...value.visibility === "private" ? { visibility: "private" } : {},
		...typeof value.owner === "string" ? { owner: value.owner } : {},
		...typeof value.icon === "string" ? { icon: value.icon } : {},
		...typeof value.createdBy === "string" ? { createdBy: value.createdBy } : {}
	};
}
const WIDGET_STATUSES = /* @__PURE__ */ new Set([
	"pending",
	"approved",
	"rejected"
]);
function normalizeRegistryEntry(value) {
	if (!isRecord$5(value)) return null;
	const status = value.status;
	if (typeof status !== "string" || !WIDGET_STATUSES.has(status)) return null;
	return {
		status,
		...typeof value.createdBy === "string" ? { createdBy: value.createdBy } : {},
		...typeof value.approvedBy === "string" ? { approvedBy: value.approvedBy } : {},
		...typeof value.approvedAt === "string" ? { approvedAt: value.approvedAt } : {}
	};
}
function normalizeWidgetsRegistry(value) {
	if (!isRecord$5(value)) return {};
	const registry = {};
	for (const [name, raw] of Object.entries(value)) {
		const entry = normalizeRegistryEntry(raw);
		if (entry) registry[name] = entry;
	}
	return registry;
}
const CAPABILITY_STATUSES = /* @__PURE__ */ new Set([
	"requested",
	"granted",
	"revoked"
]);
/** Read one capability grant defensively; drops a malformed entry (returns null). */
function normalizeCapabilityGrant(value) {
	if (!isRecord$5(value)) return null;
	const status = value.status;
	if (typeof status !== "string" || !CAPABILITY_STATUSES.has(status)) return null;
	const strings = (raw) => Array.isArray(raw) ? raw.filter((entry) => typeof entry === "string") : [];
	return {
		status,
		methods: strings(value.methods),
		streams: strings(value.streams),
		...Array.isArray(value.tools) ? { tools: strings(value.tools) } : {},
		...typeof value.toolsHash === "string" ? { toolsHash: value.toolsHash } : {},
		...Array.isArray(value.autoConfirm) ? { autoConfirm: strings(value.autoConfirm) } : {},
		...typeof value.expiresAt === "string" ? { expiresAt: value.expiresAt } : {},
		...Array.isArray(value.agents) ? { agents: strings(value.agents) } : {},
		...typeof value.description === "string" ? { description: value.description } : {},
		...typeof value.grantedBy === "string" ? { grantedBy: value.grantedBy } : {},
		...typeof value.grantedAt === "string" ? { grantedAt: value.grantedAt } : {}
	};
}
function normalizeCapabilitiesRegistry(value) {
	if (!isRecord$5(value)) return {};
	const registry = {};
	for (const [name, raw] of Object.entries(value)) {
		const grant = normalizeCapabilityGrant(raw);
		if (grant) registry[name] = grant;
	}
	return registry;
}
function normalizeWorkspace(payload) {
	const record = isRecord$5(payload) ? payload : {};
	const tabs = Array.isArray(record.tabs) ? record.tabs.map(normalizeTab).filter((tab) => tab !== null) : [];
	const prefsRecord = isRecord$5(record.prefs) ? record.prefs : {};
	const tabOrder = Array.isArray(prefsRecord.tabOrder) ? prefsRecord.tabOrder.filter((slug) => typeof slug === "string") : [];
	return {
		schemaVersion: readNumber$1(record.schemaVersion, 1),
		workspaceVersion: readNumber$1(record.workspaceVersion, 0),
		tabs,
		prefs: { tabOrder },
		widgetsRegistry: normalizeWidgetsRegistry(record.widgetsRegistry),
		capabilitiesRegistry: normalizeCapabilitiesRegistry(record.capabilitiesRegistry)
	};
}
/** The `custom:<name>` widget name, or null for builtin/unknown kinds. */
function customWidgetName(kind) {
	return kind.startsWith("custom:") ? kind.slice(7) || null : null;
}
/** Registry status for a custom widget kind, or null when not a tracked custom widget. */
function customWidgetStatus(workspace, kind) {
	const name = customWidgetName(kind);
	if (!name) return null;
	return workspace.widgetsRegistry[name]?.status ?? null;
}
/**
* Tabs in display order: honor `prefs.tabOrder` first, then any doc-order tabs the
* ordering omits, so a partial `tabOrder` still shows every tab.
*/
function orderedTabs(workspace) {
	const bySlug = new Map(workspace.tabs.map((tab) => [tab.slug, tab]));
	const ordered = [];
	const seen = /* @__PURE__ */ new Set();
	for (const slug of workspace.prefs.tabOrder) {
		const tab = bySlug.get(slug);
		if (tab && !seen.has(slug)) {
			ordered.push(tab);
			seen.add(slug);
		}
	}
	for (const tab of workspace.tabs) if (!seen.has(tab.slug)) {
		ordered.push(tab);
		seen.add(tab.slug);
	}
	return ordered;
}
function visibleTabs(workspace) {
	return orderedTabs(workspace).filter((tab) => !tab.hidden);
}
function hiddenTabs(workspace) {
	return orderedTabs(workspace).filter((tab) => tab.hidden);
}
/**
* Bucket tabs by their `createdBy` provenance for the per-agent nesting strip: a
* `user` group (also the default for an unstamped tab), a `system` group, and one
* group per distinct `agent:<id>`. Group order follows each actor's first
* appearance in the input and tab order within a group is preserved, so callers
* pass already-ordered (visible) tabs.
*/
function groupTabsByActor(tabs) {
	const groups = [];
	const byKey = /* @__PURE__ */ new Map();
	for (const tab of tabs) {
		const agentId = dashboardAgentProvenance(tab.createdBy);
		const kind = agentId ? "agent" : tab.createdBy === "system" ? "system" : "user";
		const key = kind === "agent" ? `agent:${agentId}` : kind;
		let group = byKey.get(key);
		if (!group) {
			group = {
				key,
				kind,
				agentId: kind === "agent" ? agentId : null,
				tabs: []
			};
			byKey.set(key, group);
			groups.push(group);
		}
		group.tabs.push(tab);
	}
	return groups;
}
function findTab(workspace, slug) {
	if (!slug) return;
	return workspace.tabs.find((tab) => tab.slug === slug);
}
/**
* Resolve which tab is active: prefer the requested slug if it exists and is not
* hidden; otherwise fall back to the first visible tab (or first tab of any kind).
*/
function resolveActiveSlug(workspace, requested) {
	const requestedTab = findTab(workspace, requested);
	if (requestedTab) return requestedTab.slug;
	const visible = visibleTabs(workspace);
	if (visible.length > 0) return visible[0].slug;
	return orderedTabs(workspace)[0]?.slug ?? null;
}
/** Apply a JSON pointer (RFC 6901 subset) to a value; returns the value if empty. */
function applyPointer(value, pointer) {
	if (!pointer) return value;
	const segments = pointer.split("/").slice(1).map((segment) => segment.replace(/~1/g, "/").replace(/~0/g, "~"));
	let current = value;
	for (const segment of segments) if (Array.isArray(current)) {
		const index = Number(segment);
		current = Number.isInteger(index) ? current[index] : void 0;
	} else if (isRecord$5(current)) current = current[segment];
	else return;
	return current;
}
/**
* The tab-scoped subscription table: `tabSlug -> channel -> subscriberId ->
* Subscription`. A publish resolves the publisher's tab bucket FIRST, so a message
* can only ever reach same-tab subscribers — cross-tab delivery is unreachable, not
* merely filtered out.
*/
const subscriptionsByTab = /* @__PURE__ */ new Map();
/** Monotonic source of opaque, non-guessable-by-widgets subscriber identities. */
let subscriberSeq = 0;
/** Mint a fresh broker-assigned subscriber id. Never derived from widget input. */
function nextSubscriberId() {
	subscriberSeq += 1;
	return `sub_${subscriberSeq}`;
}
/**
* Register a subscription for `subscriberId` on `(tabSlug, channel)`. Idempotent
* per `(tab, channel, subscriberId)`: re-subscribing replaces the record rather
* than stacking duplicate deliveries. Returns an unsubscribe fn scoped to exactly
* this `(tab, channel, subscriberId)` triple.
*/
function subscribe(params) {
	const { tabSlug, channel, subscriberId, deliver } = params;
	let byChannel = subscriptionsByTab.get(tabSlug);
	if (!byChannel) {
		byChannel = /* @__PURE__ */ new Map();
		subscriptionsByTab.set(tabSlug, byChannel);
	}
	let bySubscriber = byChannel.get(channel);
	if (!bySubscriber) {
		bySubscriber = /* @__PURE__ */ new Map();
		byChannel.set(channel, bySubscriber);
	}
	bySubscriber.set(subscriberId, {
		subscriberId,
		channel,
		deliver
	});
	return () => unsubscribe({
		tabSlug,
		channel,
		subscriberId
	});
}
/** Remove one subscription, pruning empty channel/tab buckets so nothing leaks. */
function unsubscribe(params) {
	const { tabSlug, channel, subscriberId } = params;
	const byChannel = subscriptionsByTab.get(tabSlug);
	const bySubscriber = byChannel?.get(channel);
	if (!bySubscriber) return;
	bySubscriber.delete(subscriberId);
	if (bySubscriber.size === 0) byChannel?.delete(channel);
	if (byChannel && byChannel.size === 0) subscriptionsByTab.delete(tabSlug);
}
/**
* Remove EVERY subscription owned by `subscriberId` on `tabSlug` (unmount teardown).
* The bridge tracks its own channels, but this is the belt-and-suspenders sweep so a
* disposed widget can never receive a dangling delivery.
*/
function unsubscribeAll(tabSlug, subscriberId) {
	const byChannel = subscriptionsByTab.get(tabSlug);
	if (!byChannel) return;
	for (const [channel, bySubscriber] of byChannel) if (bySubscriber.delete(subscriberId) && bySubscriber.size === 0) byChannel.delete(channel);
	if (byChannel.size === 0) subscriptionsByTab.delete(tabSlug);
}
/**
* Broker a publish: deliver `payload` on `channel` to every OTHER same-tab
* subscriber (the publisher, identified by `fromSubscriberId`, is excluded from its
* own broadcast). Cross-tab delivery is impossible — only the publisher's own tab
* bucket is ever consulted. Returns the number of subscribers reached (for tests).
*/
function publish(params) {
	const { tabSlug, channel, fromSubscriberId, payload } = params;
	const bySubscriber = subscriptionsByTab.get(tabSlug)?.get(channel);
	if (!bySubscriber) return 0;
	let delivered = 0;
	for (const subscription of Array.from(bySubscriber.values())) {
		if (subscription.subscriberId === fromSubscriberId) continue;
		subscription.deliver(channel, payload);
		delivered += 1;
	}
	return delivered;
}
/** Column width in pixels given the grid content width. Gaps sit between cells. */
function columnWidth(metrics) {
	return Math.max(1, (metrics.width - 132) / 12);
}
function clamp(value, min, max) {
	return Math.min(max, Math.max(min, value));
}
/** Snap a fractional column delta to whole grid units. */
function snapCells(deltaPx, unitPx) {
	if (unitPx <= 0) return 0;
	return Math.round(deltaPx / (unitPx + 12));
}
/** Clamp a rect so it stays inside the 12-column grid; height/y are unbounded below. */
function clampRect(rect) {
	const w = clamp(rect.w, 1, 12);
	const h = Math.max(1, rect.h);
	return {
		x: clamp(rect.x, 0, 12 - w),
		y: Math.max(0, rect.y),
		w,
		h
	};
}
/** Do two grid rects share any cell? Touching edges do NOT overlap. */
function rectsOverlap(a, b) {
	return a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
}
/** Rects of every widget except the one identified by `exceptId`. */
function otherRects(widgets, exceptId) {
	return widgets.filter((widget) => widget.id !== exceptId).map((widget) => widget.grid);
}
/** Does `rect` overlap any widget other than `exceptId`? */
function collides(rect, widgets, exceptId) {
	return otherRects(widgets, exceptId).some((other) => rectsOverlap(rect, other));
}
/** Begin a drag/resize gesture from a pointer-down on a widget's chrome. */
function beginDrag(params) {
	return {
		widgetId: params.widget.id,
		mode: params.mode,
		originRect: { ...params.widget.grid },
		originClientX: params.clientX,
		originClientY: params.clientY,
		ghostRect: { ...params.widget.grid },
		pointerDx: 0,
		pointerDy: 0,
		columnWidth: columnWidth(params.metrics)
	};
}
/** Advance a drag with the current pointer position; returns the snapped ghost rect. */
function updateDrag(drag, clientX, clientY) {
	drag.pointerDx = clientX - drag.originClientX;
	drag.pointerDy = clientY - drag.originClientY;
	const rowUnit = 56;
	const deltaCols = snapCells(clientX - drag.originClientX, drag.columnWidth);
	const deltaRows = snapCells(clientY - drag.originClientY, rowUnit);
	const clamped = clampRect(drag.mode === "move" ? {
		x: drag.originRect.x + deltaCols,
		y: drag.originRect.y + deltaRows,
		w: drag.originRect.w,
		h: drag.originRect.h
	} : {
		x: drag.originRect.x,
		y: drag.originRect.y,
		w: drag.originRect.w + deltaCols,
		h: drag.originRect.h + deltaRows
	});
	drag.ghostRect = clamped;
	return clamped;
}
/**
* Resolve where a dropped widget lands. Overlapping drops are rejected; the
* nearest collision-free slot to the requested position is returned instead.
* Returns null only if the grid genuinely has no free slot for the widget's size
* (defensive; the grid is unbounded downward so this is unreachable in practice).
*/
function resolveDrop(params) {
	const requested = clampRect(params.requested);
	if (!collides(requested, params.widgets, params.widgetId)) return requested;
	return nearestFreeSlot(requested, params.widgets, params.widgetId);
}
/**
* Search outward from the requested position for the closest slot that fits
* `requested`'s size without colliding. The grid grows downward, so a fit is
* always found within a bounded number of rows.
*/
function nearestFreeSlot(requested, widgets, widgetId) {
	const w = clamp(requested.w, 1, 12);
	const h = Math.max(1, requested.h);
	const maxX = 12 - w;
	const occupiedRows = otherRects(widgets, widgetId).reduce((max, rect) => Math.max(max, rect.y + rect.h), 0);
	const maxY = Math.max(requested.y, occupiedRows) + h;
	let best = null;
	for (let y = 0; y <= maxY; y += 1) {
		for (let x = 0; x <= maxX; x += 1) {
			const candidate = {
				x,
				y,
				w,
				h
			};
			if (collides(candidate, widgets, widgetId)) continue;
			const distance = Math.abs(x - requested.x) + Math.abs(y - requested.y);
			if (!best || distance < best.distance) best = {
				rect: candidate,
				distance
			};
		}
		if (best && y >= requested.y) break;
	}
	return best?.rect ?? null;
}
/** CSS grid-column/grid-row shorthand for a rect (1-based grid lines). */
function gridPlacementStyle(rect) {
	return [`grid-column: ${rect.x + 1} / span ${rect.w}`, `grid-row: ${rect.y + 1} / span ${rect.h}`].join("; ");
}
/** Total rows a set of widgets spans (for sizing the grid's min-height). */
function gridRowCount(widgets) {
	return widgets.reduce((max, widget) => Math.max(max, widget.grid.y + widget.grid.h), 0);
}
/** Nudge a rect by keyboard for the a11y move/resize fallback. */
function nudgeRect(rect, mode, direction) {
	const step = 1;
	if (mode === "move") {
		const dx = direction === "left" ? -1 : direction === "right" ? step : 0;
		const dy = direction === "up" ? -1 : direction === "down" ? step : 0;
		return clampRect({
			...rect,
			x: rect.x + dx,
			y: rect.y + dy
		});
	}
	const dw = direction === "left" ? -1 : direction === "right" ? step : 0;
	const dh = direction === "up" ? -1 : direction === "down" ? step : 0;
	return clampRect({
		...rect,
		w: rect.w + dw,
		h: rect.h + dh
	});
}
const PENDING_STATUS = "pending";
function isRecord$3$1(value) {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
/**
* Pull the strict workspace doc out of a `dashboard.workspace.get` payload. The
* host responds `{ doc, workspaceVersion }`; `.workspace` and the bare payload are
* accepted as fallbacks so export is robust to the response envelope.
*/
function workspaceDocFromPayload(payload) {
	if (isRecord$3$1(payload)) {
		if (isRecord$3$1(payload.doc)) return payload.doc;
		if (isRecord$3$1(payload.workspace)) return payload.workspace;
		return payload;
	}
	return {};
}
/** Timestamped download filename, e.g. `dashboard-workspace-2026-07-08T12-00-00-000Z.json`. */
function workspaceExportFilename(now = /* @__PURE__ */ new Date()) {
	return `dashboard-workspace-${now.toISOString().replace(/[:.]/g, "-")}.json`;
}
/** The `custom:<name>` name for a widget kind, or null for builtin/unknown kinds. */
function customName(kind) {
	if (typeof kind !== "string" || !kind.startsWith("custom:")) return null;
	return kind.slice(7) || null;
}
/** Every custom-widget name referenced by the tabs' widgets. */
function customWidgetNames(tabs) {
	const names = /* @__PURE__ */ new Set();
	if (!Array.isArray(tabs)) return names;
	for (const tab of tabs) {
		const widgets = isRecord$3$1(tab) && Array.isArray(tab.widgets) ? tab.widgets : [];
		for (const widget of widgets) {
			const name = isRecord$3$1(widget) ? customName(widget.kind) : null;
			if (name) names.add(name);
		}
	}
	return names;
}
/** Keep only the registry entries whose custom widget still appears in `tabs`. */
function pruneRegistry(tabs, registry) {
	if (!isRecord$3$1(registry)) return {};
	const referenced = customWidgetNames(tabs);
	const pruned = {};
	for (const [name, entry] of Object.entries(registry)) if (referenced.has(name)) pruned[name] = entry;
	return pruned;
}
/**
* Build the export doc: the full workspace, or a subset filtered to `slugs`. A
* subset prunes `prefs.tabOrder` to the kept slugs and the registry to the custom
* widgets those tabs still reference, so the result stays a valid WorkspaceDoc.
*/
function buildWorkspaceExportDoc(doc, options = {}) {
	const clone = structuredClone(doc);
	const slugs = options.slugs;
	if (!slugs || slugs.length === 0) return clone;
	const keep = new Set(slugs);
	const tabs = Array.isArray(clone.tabs) ? clone.tabs.filter((tab) => isRecord$3$1(tab) && keep.has(tab.slug)) : [];
	clone.tabs = tabs;
	const prefs = isRecord$3$1(clone.prefs) ? clone.prefs : {};
	const tabOrder = Array.isArray(prefs.tabOrder) ? prefs.tabOrder : [];
	clone.prefs = {
		...prefs,
		tabOrder: tabOrder.filter((slug) => typeof slug === "string" && keep.has(slug))
	};
	clone.widgetsRegistry = pruneRegistry(tabs, clone.widgetsRegistry);
	return clone;
}
/** Serialize the export doc as pretty JSON with a trailing newline (matches the store). */
function serializeWorkspaceExport(doc, options = {}) {
	return `${JSON.stringify(buildWorkspaceExportDoc(doc, options), null, 2)}\n`;
}
/** Parse an imported file into JSON, surfacing a friendly error on malformed input. */
function parseWorkspaceImport(text) {
	try {
		return JSON.parse(text);
	} catch {
		throw new Error("Import file is not valid JSON.");
	}
}
function toPendingEntry(entry) {
	const createdBy = isRecord$3$1(entry) && typeof entry.createdBy === "string" ? entry.createdBy : "user";
	return {
		status: PENDING_STATUS,
		createdBy
	};
}
/**
* Coerce every custom widget referenced by an imported doc to `pending` so the
* approval gate runs before it can mount — an import NEVER auto-approves a custom
* widget. Forces pending unconditionally because an imported workspace is foreign,
* untrusted authoring. Structural validation is left to the server
* (`dashboard.workspace.replace`).
*/
function sanitizeImportedWorkspace(parsed) {
	if (!isRecord$3$1(parsed)) throw new Error("Import file must be a workspace object.");
	const doc = structuredClone(parsed);
	const registryInput = isRecord$3$1(doc.widgetsRegistry) ? doc.widgetsRegistry : {};
	const registry = {};
	for (const [name, entry] of Object.entries(registryInput)) registry[name] = toPendingEntry(entry);
	for (const name of customWidgetNames(doc.tabs)) registry[name] ??= {
		status: PENDING_STATUS,
		createdBy: "user"
	};
	doc.widgetsRegistry = registry;
	const capsInput = isRecord$3$1(doc.capabilitiesRegistry) ? doc.capabilitiesRegistry : {};
	const caps = {};
	for (const [name, entry] of Object.entries(capsInput)) if (isRecord$3$1(entry)) {
		const { grantedBy: _grantedBy, grantedAt: _grantedAt, autoConfirm: _autoConfirm, expiresAt: _expiresAt, agents: _agents, ...rest } = entry;
		caps[name] = {
			...rest,
			status: "requested"
		};
	}
	doc.capabilitiesRegistry = caps;
	return doc;
}
/** The one-liner (grant `description`) for a connector's approval card. */
function recipeGrantDescription(grant) {
	const reason = grant.reason?.trim();
	return reason && reason.length > 0 ? reason.slice(0, 200) : void 0;
}
/**
* Build the workspace doc a recipe installs: the recipe's `doc` with its
* `capabilitiesRegistry` REPLACED by the grants the `grantsManifest` declares, each
* `requested`. The result is NOT yet re-pended — pass it through
* `sanitizeImportedWorkspace` (as `buildRecipeImportDoc` does) so it travels the same
* seam every imported board does. `toolsHash` is deliberately omitted: the broker
* reconciles a `requested` grant's tool surface to the connector's live manifest on its
* next refresh, so the recipe declares INTENT and the host owns the authoritative hash.
*/
function buildRecipeInstallDoc(recipe) {
	const doc = structuredClone(recipe.doc);
	const caps = {};
	for (const [connector, grant] of Object.entries(recipe.grantsManifest)) {
		const description = recipeGrantDescription(grant);
		const tools = (grant.tools ?? []).map((tool) => tool.id);
		caps[connector] = {
			status: "requested",
			methods: grant.methods ?? [],
			streams: grant.streams ?? [],
			...tools.length > 0 ? { tools } : {},
			...description !== void 0 ? { description } : {}
		};
	}
	doc.capabilitiesRegistry = caps;
	return doc;
}
/**
* The doc to hand `dashboard.workspace.replace` when installing a recipe: the recipe's
* board with its manifest grants merged in, then run through the SAME
* `sanitizeImportedWorkspace` re-pend as any imported workspace. Install therefore
* inherits every import guarantee — pending widgets, requested grants, stripped
* auto-run/TTL — for free, and can never grant.
*/
function buildRecipeImportDoc(recipe) {
	return sanitizeImportedWorkspace(buildRecipeInstallDoc(recipe));
}
/** Hard client-side cap on a fetched bundle; the host re-checks server-side. */
const GALLERY_BUNDLE_MAX_BYTES = 512 * 1024;
/** Hard client-side cap on a fetched recipe bundle (a doc can be large but bounded). */
const GALLERY_RECIPE_MAX_BYTES = 512 * 1024;
/** Hard client-side cap on a fetched registry index. */
const GALLERY_INDEX_MAX_BYTES = 256 * 1024;
const CUSTOM_WIDGET_NAME_PATTERN$1 = /^[A-Za-z0-9._-]{1,64}$/;
function isRecord$1$4(value) {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
/** UTF-8 byte length of a string (the cap unit; enforced by the host fetch layer). */
function galleryByteLength(text) {
	return new TextEncoder().encode(text).length;
}
/**
* Parse a registry `index.json` text (CLIENT-fetched). Accepts either a bare array
* of entries or `{ widgets: [...] }`. Relative `manifestUrl`s resolve against
* `indexUrl`. Malformed entries are dropped rather than throwing.
*/
function parseGalleryIndex(text, indexUrl) {
	let parsed;
	try {
		parsed = JSON.parse(text);
	} catch {
		throw new Error("The gallery index is not valid JSON.");
	}
	const rawList = Array.isArray(parsed) ? parsed : isRecord$1$4(parsed) && Array.isArray(parsed.widgets) ? parsed.widgets : null;
	if (!rawList) throw new Error("The gallery index must be a list of widgets.");
	const entries = [];
	for (const raw of rawList) {
		if (!isRecord$1$4(raw)) continue;
		const name = typeof raw.name === "string" ? raw.name.trim() : "";
		const manifestUrlRaw = typeof raw.manifestUrl === "string" ? raw.manifestUrl.trim() : "";
		if (!CUSTOM_WIDGET_NAME_PATTERN$1.test(name) || !manifestUrlRaw) continue;
		let manifestUrl;
		try {
			manifestUrl = new URL(manifestUrlRaw, indexUrl).toString();
		} catch {
			continue;
		}
		entries.push({
			name,
			description: typeof raw.description === "string" ? raw.description : "",
			manifestUrl
		});
	}
	return entries;
}
function readCapabilities(value) {
	if (!Array.isArray(value)) return [];
	return value.filter((cap) => cap === "data:read" || cap === "prompt:send");
}
function readBindingIds(value) {
	if (!Array.isArray(value)) return [];
	return value.map((binding) => isRecord$1$4(binding) && typeof binding.id === "string" ? binding.id : null).filter((id) => id !== null);
}
/**
* Parse a widget-bundle text (CLIENT-fetched) and shape-check it enough to preview
* and hand to the install RPC. The bundle is `{ manifest, files }`; the manifest is
* the widget's `widget.json` object. Authoritative manifest validation happens
* server-side on install.
*/
function parseWidgetBundle(text) {
	let parsed;
	try {
		parsed = JSON.parse(text);
	} catch {
		throw new Error("The widget bundle is not valid JSON.");
	}
	if (!isRecord$1$4(parsed) || !isRecord$1$4(parsed.manifest) || !isRecord$1$4(parsed.files)) throw new Error("The widget bundle must be an object with `manifest` and `files`.");
	const manifest = parsed.manifest;
	const name = typeof manifest.name === "string" ? manifest.name.trim() : "";
	if (!CUSTOM_WIDGET_NAME_PATTERN$1.test(name)) throw new Error("The widget bundle manifest has an invalid name.");
	const files = {};
	for (const [key, content] of Object.entries(parsed.files)) {
		if (typeof content !== "string") throw new Error("Every widget bundle file must be text.");
		files[key] = content;
	}
	return {
		name,
		title: typeof manifest.title === "string" ? manifest.title : name,
		capabilities: readCapabilities(manifest.capabilities),
		bindingIds: readBindingIds(manifest.bindings),
		manifest,
		files
	};
}
const RECIPE_NAME_PATTERN = /^[A-Za-z0-9._-]{1,64}$/;
/**
* Parse a registry `index.json` text's `recipes` array (CLIENT-fetched), sibling of the
* widget entries. Relative `manifestUrl`s resolve against `indexUrl`; malformed entries
* are dropped rather than throwing. An index with no `recipes` key yields `[]`.
*/
function parseRecipeIndex(text, indexUrl) {
	let parsed;
	try {
		parsed = JSON.parse(text);
	} catch {
		throw new Error("The gallery index is not valid JSON.");
	}
	const rawList = isRecord$1$4(parsed) && Array.isArray(parsed.recipes) ? parsed.recipes : null;
	if (!rawList) return [];
	const entries = [];
	for (const raw of rawList) {
		if (!isRecord$1$4(raw)) continue;
		const name = typeof raw.name === "string" ? raw.name.trim() : "";
		const manifestUrlRaw = typeof raw.manifestUrl === "string" ? raw.manifestUrl.trim() : "";
		if (!RECIPE_NAME_PATTERN.test(name) || !manifestUrlRaw) continue;
		let manifestUrl;
		try {
			manifestUrl = new URL(manifestUrlRaw, indexUrl).toString();
		} catch {
			continue;
		}
		const connectors = Array.isArray(raw.connectors) ? raw.connectors.filter((c) => typeof c === "string") : [];
		entries.push({
			name,
			title: typeof raw.title === "string" && raw.title ? raw.title : name,
			description: typeof raw.description === "string" ? raw.description : "",
			manifestUrl,
			connectors
		});
	}
	return entries;
}
/**
* Parse + fully validate a recipe bundle text (CLIENT-fetched). Unlike a widget bundle
* (whose manifest is authoritatively validated server-side on install), a recipe is pure
* data applied through `dashboard.workspace.replace`, so it is validated in full HERE with
* the shared `validateRecipe` — the same guard the honesty gate runs over every shipped
* recipe. Throws a friendly error on malformed input.
*/
function parseRecipeBundle(text) {
	let parsed;
	try {
		parsed = JSON.parse(text);
	} catch {
		throw new Error("The recipe bundle is not valid JSON.");
	}
	try {
		return validateRecipe(parsed);
	} catch (err) {
		throw new Error(`The recipe bundle is invalid: ${err instanceof Error ? err.message : String(err)}`);
	}
}
function widgetProps(widget) {
	return widget.props ?? {};
}
function isRecord$4(value) {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
/** Coerce a possibly-string numeric field to a finite number, else undefined. */
function toFiniteNumber(value) {
	if (typeof value === "number") return Number.isFinite(value) ? value : void 0;
	if (typeof value === "string" && value.trim()) {
		const parsed = Number(value);
		return Number.isFinite(parsed) ? parsed : void 0;
	}
}
/** Named metrics selectable from a structured binding payload via `props.metric`. */
function selectMetric(value, metric) {
	if (!isRecord$4(value)) return;
	const totals = isRecord$4(value.totals) ? value.totals : void 0;
	switch (metric) {
		case "todayCost": return totals?.totalCost ?? value.totalCost;
		case "todayTokens": return totals?.totalTokens ?? value.totalTokens;
		default: return value[metric];
	}
}
function formatStatValue(value, format) {
	if (value === void 0 || value === null) return null;
	const numeric = toFiniteNumber(value);
	if (format === "usd" && numeric !== void 0) return new Intl.NumberFormat("en-US", {
		style: "currency",
		currency: "USD"
	}).format(numeric);
	if (format === "percent" && numeric !== void 0) return new Intl.NumberFormat("en-US", {
		style: "percent",
		maximumFractionDigits: 1
	}).format(numeric);
	if ((format === "int" || format === "integer") && numeric !== void 0) return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(numeric);
	if (typeof value === "string") return value;
	if (numeric !== void 0) return new Intl.NumberFormat("en-US").format(numeric);
	return JSON.stringify(value);
}
function mapStatCard(widget, value) {
	const props = widgetProps(widget);
	const metric = typeof props.metric === "string" ? props.metric : null;
	const selected = metric ? selectMetric(value, metric) : value;
	const resolved = selected !== void 0 ? selected : props.value;
	const label = typeof props.label === "string" ? props.label : widget.title;
	const dedupedLabel = label && label !== widget.title ? label : null;
	return {
		display: formatStatValue(resolved, props.format),
		label: dedupedLabel
	};
}
function mapMarkdownSource(widget, value) {
	const props = widgetProps(widget);
	if (typeof value === "string") return value;
	if (typeof props.markdown === "string") return props.markdown;
	if (typeof props.text === "string") return props.text;
	return "";
}
const DEFAULT_ROW_LIMIT = 8;
/** Pull an array of row records out of the binding value or `props.rows`. */
function resolveRows(widget, value) {
	return (Array.isArray(value) ? value : isRecord$4(value) && Array.isArray(value.rows) ? value.rows : Array.isArray(widgetProps(widget).rows) ? widgetProps(widget).rows : []).filter(isRecord$4);
}
function resolveColumns(widget, rows) {
	const declared = widgetProps(widget).columns;
	if (Array.isArray(declared)) {
		const picked = declared.filter((entry) => typeof entry === "string");
		if (picked.length > 0) return picked;
	}
	return rows.length > 0 ? Object.keys(rows[0]) : [];
}
function rowLimit(widget) {
	const raw = widgetProps(widget).limit;
	return typeof raw === "number" && Number.isFinite(raw) && raw > 0 ? Math.min(Math.trunc(raw), 100) : DEFAULT_ROW_LIMIT;
}
function mapTable(widget, value) {
	const all = resolveRows(widget, value);
	const limit = rowLimit(widget);
	const rows = all.slice(0, limit);
	return {
		columns: resolveColumns(widget, rows),
		rows,
		shown: rows.length,
		total: all.length
	};
}
const DEFAULT_LIMIT$5 = 6;
/** Live-run predicate: a non-`running` status is inactive; else fall back to `hasActiveRun`. */
function isSessionRunActive$1(state) {
	if (state.status && state.status !== "running") return false;
	if (typeof state.hasActiveRun === "boolean") return state.hasActiveRun;
	return state.status === "running";
}
function rowLabel$1(row, key) {
	const display = row.displayName ?? row.label ?? row.subject ?? row.channel;
	return typeof display === "string" && display.trim() ? display : key;
}
function mapSessions(widget, value) {
	const raw = Array.isArray(value) ? value : isRecord$4(value) && Array.isArray(value.sessions) ? value.sessions : [];
	const limitProp = toFiniteNumber(widgetProps(widget).limit);
	const limit = limitProp && limitProp > 0 ? Math.trunc(limitProp) : DEFAULT_LIMIT$5;
	const records = raw.filter(isRecord$4);
	return {
		rows: records.map((row) => {
			const key = typeof row.key === "string" ? row.key : "";
			return {
				key,
				label: rowLabel$1(row, key),
				active: isSessionRunActive$1({
					hasActiveRun: typeof row.hasActiveRun === "boolean" ? row.hasActiveRun : void 0,
					status: typeof row.status === "string" ? row.status : void 0
				}),
				updatedAt: toFiniteNumber(row.updatedAt) ?? null
			};
		}).filter((row) => row.key).slice(0, limit),
		total: records.length
	};
}
function mapUsage(_widget, value) {
	const totals = isRecord$4(value) && isRecord$4(value.totals) ? value.totals : {};
	return {
		cost: toFiniteNumber(totals.totalCost) ?? 0,
		tokens: toFiniteNumber(totals.totalTokens) ?? 0,
		days: isRecord$4(value) ? toFiniteNumber(value.days) ?? null : null
	};
}
const DEFAULT_LIMIT$4 = 8;
function jobStatus(state) {
	if (!state) return null;
	const status = state.lastRunStatus ?? state.lastStatus;
	return typeof status === "string" ? status : null;
}
function mapCron(widget, value) {
	const raw = isRecord$4(value) && Array.isArray(value.jobs) ? value.jobs : [];
	const limitProp = toFiniteNumber(widgetProps(widget).limit);
	const limit = limitProp && limitProp > 0 ? Math.trunc(limitProp) : DEFAULT_LIMIT$4;
	const records = raw.filter(isRecord$4);
	return {
		jobs: records.map((job) => {
			const state = isRecord$4(job.state) ? job.state : void 0;
			return {
				id: typeof job.id === "string" ? job.id : "",
				name: typeof job.name === "string" && job.name.trim() ? job.name : job.id || "",
				enabled: job.enabled !== false,
				nextRunAtMs: state ? toFiniteNumber(state.nextRunAtMs) ?? null : null,
				lastStatus: jobStatus(state)
			};
		}).filter((job) => job.id).slice(0, limit),
		total: records.length
	};
}
const DEFAULT_LIMIT$3 = 8;
const HEALTHY_IDLE_SECONDS = 120;
function instanceId(entry) {
	const candidate = entry.instanceId ?? entry.host ?? entry.ip ?? entry.deviceFamily;
	return typeof candidate === "string" && candidate.trim() ? candidate : "";
}
function instanceDetail(entry) {
	const parts = [
		entry.mode,
		entry.platform,
		entry.version
	].filter((part) => typeof part === "string" && part.trim().length > 0);
	return parts.length > 0 ? parts.join(" · ") : null;
}
function mapInstances(widget, value) {
	const raw = Array.isArray(value) ? value : isRecord$4(value) && Array.isArray(value.presence) ? value.presence : isRecord$4(value) && Array.isArray(value.nodes) ? value.nodes : [];
	const limitProp = toFiniteNumber(widgetProps(widget).limit);
	const limit = limitProp && limitProp > 0 ? Math.trunc(limitProp) : DEFAULT_LIMIT$3;
	const records = raw.filter(isRecord$4);
	return {
		instances: records.map((entry) => {
			const lastInputSeconds = toFiniteNumber(entry.lastInputSeconds);
			return {
				id: instanceId(entry),
				detail: instanceDetail(entry),
				healthy: lastInputSeconds === void 0 || lastInputSeconds <= HEALTHY_IDLE_SECONDS,
				lastInputMs: lastInputSeconds !== void 0 ? lastInputSeconds * 1e3 : null
			};
		}).filter((entry) => entry.id).slice(0, limit),
		total: records.length
	};
}
const DEFAULT_LIMIT$2 = 20;
/** Truncate a summary/error line to a bounded length with an ellipsis. */
function clampText$1(value, max = 120) {
	if (value.length <= max) return value;
	return `${value.slice(0, Math.max(0, max - 1))}…`;
}
function entryTitle(entry) {
	const name = entry.jobName ?? entry.jobId ?? entry.action;
	return typeof name === "string" && name.trim() ? name : "run";
}
function mapActivity(widget, value) {
	const raw = isRecord$4(value) && Array.isArray(value.entries) ? value.entries : [];
	const limitProp = toFiniteNumber(widgetProps(widget).limit);
	const limit = limitProp && limitProp > 0 ? Math.trunc(limitProp) : DEFAULT_LIMIT$2;
	const records = raw.filter(isRecord$4);
	return {
		entries: records.map((entry) => ({
			ts: toFiniteNumber(entry.ts) ?? null,
			title: entryTitle(entry),
			detail: typeof entry.summary === "string" && entry.summary.trim() ? clampText$1(entry.summary, 120) : typeof entry.error === "string" && entry.error.trim() ? clampText$1(entry.error, 120) : null,
			status: typeof entry.status === "string" ? entry.status : null
		})).slice(0, limit),
		total: records.length
	};
}
/**
* Resolve `rawUrl` against the embed policy. Relative URLs and same-origin
* absolute URLs are internal and always allowed. Absolute http(s) URLs to a
* different origin are external and require `allowExternalEmbedUrls`. Any other
* scheme (javascript:, data:, file:, …) is rejected outright.
*/
function evaluateEmbedUrl(rawUrl, policy, origin) {
	if (typeof rawUrl !== "string" || !rawUrl.trim()) return { status: "missing" };
	const url = rawUrl.trim();
	const ambientOrigin = globalThis.location?.origin;
	const base = origin ?? ambientOrigin;
	let parsed;
	try {
		parsed = base ? new URL(url, base) : new URL(url);
	} catch {
		return {
			status: "ok",
			url,
			external: false
		};
	}
	if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return {
		status: "blocked",
		reason: "scheme",
		url
	};
	const external = base ? parsed.origin !== new URL(base).origin : true;
	if (external && !policy.allowExternalEmbedUrls) return {
		status: "blocked",
		reason: "external",
		url
	};
	return {
		status: "ok",
		url,
		external
	};
}
const CHART_TYPES = [
	"line",
	"bar",
	"area",
	"sparkline",
	"gauge"
];
const DEFAULT_TYPE = "line";
/** Pull a numeric y from a point-like entry (`y`, else `value`). */
function pointValue(entry) {
	if (typeof entry === "number") return Number.isFinite(entry) ? entry : void 0;
	if (isRecord$4(entry)) return toFiniteNumber(entry.y) ?? toFiniteNumber(entry.value);
}
/** Coerce the tolerant binding value into a plain, finite `number[]`. */
function normalizeSeries(value) {
	const raw = Array.isArray(value) ? value : isRecord$4(value) && Array.isArray(value.points) ? value.points : [];
	const out = [];
	for (const entry of raw) {
		const n = pointValue(entry);
		if (n !== void 0) out.push(n);
	}
	return out;
}
function resolveType(props) {
	const raw = props.type;
	return typeof raw === "string" && CHART_TYPES.includes(raw) ? raw : DEFAULT_TYPE;
}
function mapChart(widget, value) {
	const props = widgetProps(widget);
	const values = normalizeSeries(value);
	const min = values.length ? Math.min(...values) : 0;
	const max = values.length ? Math.max(...values) : 0;
	return {
		type: resolveType(props),
		values,
		min,
		max,
		detail: props.detail === true,
		label: props.label === true
	};
}
/** Coerce a persisted state blob to the editable text. Stored blob is the raw string. */
function notesTextFromState(state) {
	if (typeof state === "string") return state;
	return "";
}
const SLOT_PATTERN = /\{([A-Za-z0-9_]+)\}/g;
const FIELD_TYPES = /* @__PURE__ */ new Set([
	"text",
	"number",
	"select"
]);
/** Defensively parse one field descriptor from untyped props, or null when malformed. */
function mapField(value) {
	if (!isRecord$4(value)) return null;
	const { name, label, type } = value;
	if (typeof name !== "string" || !name || typeof label !== "string" || !label) return null;
	if (typeof type !== "string" || !FIELD_TYPES.has(type)) return null;
	const options = type === "select" && Array.isArray(value.options) ? value.options.filter((option) => typeof option === "string") : void 0;
	if (type === "select" && (!options || options.length === 0)) return null;
	const maxLength = typeof value.maxLength === "number" && Number.isInteger(value.maxLength) && value.maxLength > 0 ? value.maxLength : void 0;
	return {
		name,
		label,
		type,
		...options ? { options } : {},
		...maxLength !== void 0 ? { maxLength } : {}
	};
}
/** Read one string→string mapping defensively (tool-mode `argsFrom`); drops non-string values. */
function mapArgsFrom(value) {
	if (!isRecord$4(value)) return {};
	const out = {};
	for (const [argName, fieldName] of Object.entries(value)) if (typeof fieldName === "string") out[argName] = fieldName;
	return out;
}
/** Read the action-form view model from a widget's props (defensive; schema is the gate). */
function mapActionForm(widget) {
	const props = widgetProps(widget);
	const template = typeof props.template === "string" ? props.template : "";
	const fields = Array.isArray(props.fields) ? props.fields.map(mapField).filter((field) => field !== null) : [];
	const buttonLabel = typeof props.buttonLabel === "string" ? props.buttonLabel : null;
	if ((props.mode === "tool" ? "tool" : "prompt") !== "tool") return {
		template,
		fields,
		buttonLabel,
		mode: "prompt",
		connector: null,
		tool: null,
		argsFrom: null
	};
	return {
		template,
		fields,
		buttonLabel,
		mode: "tool",
		connector: typeof props.connector === "string" ? props.connector : null,
		tool: typeof props.tool === "string" ? props.tool : null,
		argsFrom: mapArgsFrom(props.argsFrom)
	};
}
/** Type + length cap for one field's raw string value. Non-numeric numbers and out-of-set selects collapse to "". */
function coerceFieldValue(field, raw) {
	const cap = field.maxLength && field.maxLength > 0 ? field.maxLength : 200;
	if (field.type === "number") {
		const trimmed = raw.trim();
		return trimmed && Number.isFinite(Number(trimmed)) ? trimmed.slice(0, cap) : "";
	}
	if (field.type === "select") return field.options?.includes(raw) ? raw : "";
	return raw.slice(0, cap);
}
/**
* Interpolate declared field values into the authored template in a SINGLE pass.
* Only `{slot}` tokens that name a declared field are replaced; the replacement
* text is inserted literally (function replacer) and never re-scanned, so a value
* containing `{...}` cannot expand. Unknown slots are left verbatim.
*/
function buildActionFormPrompt(model, values) {
	const byName = new Map(model.fields.map((field) => [field.name, field]));
	return model.template.replace(SLOT_PATTERN, (match, name) => {
		const field = byName.get(name);
		if (!field) return match;
		return coerceFieldValue(field, values[name] ?? "");
	});
}
/**
* Build the tool-mode argument object from a `tool`-mode form's submitted field values.
* Each `argsFrom` entry maps a tool ARGUMENT name to a declared FIELD name; the field's
* raw value is typed + length-capped (`coerceFieldValue`) before it lands as an argument.
* There is NO template interpolation here — the fields ARE the args (the prompt path is
* unrelated). An entry naming an undeclared field, or a non-`tool` model, is skipped, so
* an argument can never carry an undeclared value.
*/
function buildActionToolArgs(model, values) {
	const byName = new Map(model.fields.map((field) => [field.name, field]));
	const args = {};
	for (const [argName, fieldName] of Object.entries(model.argsFrom ?? {})) {
		const field = byName.get(fieldName);
		if (field) args[argName] = coerceFieldValue(field, values[fieldName] ?? "");
	}
	return args;
}
/**
* Read the action-button view model from a widget's props (defensive; the schema
* `validateActionButtonProps` gate is the real bound). A malformed connector/tool
* yields empty strings so the renderer degrades to an inert placeholder rather than
* invoking against a bad ref.
*/
function mapActionButton(widget) {
	const props = widgetProps(widget);
	return {
		connector: typeof props.connector === "string" ? props.connector : "",
		tool: typeof props.tool === "string" ? props.tool : "",
		args: isRecord$4(props.args) ? props.args : null,
		label: typeof props.label === "string" ? props.label : null
	};
}
const PREVIEW_VIEWPORTS$1 = [
	"desktop",
	"tablet",
	"mobile"
];
/** Resolve the initial viewport from `props.defaultViewport`, defaulting to desktop. */
function mapPreviewViewport(widget) {
	const raw = widgetProps(widget).defaultViewport;
	return typeof raw === "string" && PREVIEW_VIEWPORTS$1.includes(raw) ? raw : "desktop";
}
const DEFAULT_LIMIT$1 = 8;
/** Live-run predicate: a non-`running` status is inactive; else fall back to `hasActiveRun`. */
function isSessionRunActive(state) {
	if (state.status && state.status !== "running") return false;
	if (typeof state.hasActiveRun === "boolean") return state.hasActiveRun;
	return state.status === "running";
}
/** Truncate to `max` characters, appending an ellipsis when clipped. */
function clampText(text, max) {
	return text.length <= max ? text : `${text.slice(0, Math.max(0, max - 1))}…`;
}
function rowLabel(row, key) {
	const display = row.displayName ?? row.label ?? row.subject ?? row.channel;
	return typeof display === "string" && display.trim() ? display : key;
}
/** Current task/objective for the row: the active goal objective, if any. */
function rowTask(row) {
	const goal = isRecord$4(row.goal) ? row.goal : void 0;
	const objective = goal && typeof goal.objective === "string" ? goal.objective.trim() : "";
	return objective ? clampText(objective, 100) : null;
}
/** Fractional run progress from a goal's token budget, clamped to [0,1]. */
function rowProgress(row) {
	const goal = isRecord$4(row.goal) ? row.goal : void 0;
	if (!goal) return null;
	const used = toFiniteNumber(goal.tokensUsed);
	const budget = toFiniteNumber(goal.tokenBudget);
	if (used === void 0 || budget === void 0 || budget <= 0) return null;
	return Math.min(1, Math.max(0, used / budget));
}
function mapAgentStatus(widget, value) {
	const raw = Array.isArray(value) ? value : isRecord$4(value) && Array.isArray(value.sessions) ? value.sessions : [];
	const limitProp = toFiniteNumber(widgetProps(widget).limit);
	const limit = limitProp && limitProp > 0 ? Math.trunc(limitProp) : DEFAULT_LIMIT$1;
	const mapped = raw.filter(isRecord$4).map((row) => {
		const key = typeof row.key === "string" ? row.key : "";
		return {
			key,
			label: rowLabel(row, key),
			active: isSessionRunActive({
				hasActiveRun: typeof row.hasActiveRun === "boolean" ? row.hasActiveRun : void 0,
				status: typeof row.status === "string" ? row.status : void 0
			}),
			task: rowTask(row),
			progress: rowProgress(row)
		};
	}).filter((row) => row.key);
	const activeCount = mapped.filter((row) => row.active).length;
	return {
		rows: mapped.slice(0, limit),
		activeCount,
		total: mapped.length
	};
}
const DEFAULT_LIMIT = 8;
/** Map an approvals widget's UI decision to the registry decision `approveWidget` takes. */
function toWidgetApprovalDecision(decision) {
	return decision === "approve" ? "approved" : "rejected";
}
/**
* Derive the pending-widget-approval source from the workspace registry, wiring
* each decision through `resolve` (the view passes `approveWidget`). Pure so the
* view and tests build the identical source.
*/
function buildWidgetApprovalsSource(workspace, resolve) {
	return {
		pending: Object.entries(workspace.widgetsRegistry).filter(([, entry]) => entry.status === "pending").map(([name, entry]) => ({
			id: name,
			kind: "widget",
			title: name,
			requestedBy: dashboardAgentProvenance(entry.createdBy)
		})),
		onDecide: (item, decision) => resolve(item.id, toWidgetApprovalDecision(decision))
	};
}
/**
* The combined pending-approval source: agent-authored WIDGETS, data/tool CAPABILITY
* requests (SPEC §17), and server-enforced pending ACTIONS (SPEC §18). Widget
* decisions route through `resolveWidget` (`approveWidget`); capability decisions
* through `resolveCapability` (`approveCapability`) — carrying the operator's partial
* `tools` subset when they ticked one (§17.1); action decisions through the optional
* `actions.resolve` (`dashboard.action.confirm`/`deny`). Any board with an `approvals`
* widget then surfaces all three — the single operator queue.
*/
function buildApprovalsSource(workspace, resolveWidget, resolveCapability, actions) {
	const widgets = buildWidgetApprovalsSource(workspace, resolveWidget).pending;
	const grants = Object.entries(workspace.capabilitiesRegistry ?? {});
	const reachOf = (grant) => {
		const tools = grant.tools ?? [];
		const reach = [
			grant.methods.length ? `${grant.methods.length} read${grant.methods.length === 1 ? "" : "s"}` : null,
			grant.streams.length ? `${grant.streams.length} stream${grant.streams.length === 1 ? "" : "s"}` : null,
			tools.length ? `${tools.length} tool${tools.length === 1 ? "" : "s"}` : null
		].filter(Boolean);
		return grant.description ?? (reach.length ? `wants ${reach.join(" + ")}` : "data access");
	};
	const requested = grants.filter(([, grant]) => grant.status === "requested").map(([name, grant]) => ({
		id: name,
		kind: "capability",
		title: name,
		requestedBy: null,
		detail: reachOf(grant),
		...(grant.tools ?? []).length ? { tools: grant.tools } : {}
	}));
	const granted = grants.filter(([, grant]) => grant.status === "granted" && ((grant.tools ?? []).length > 0 || grant.expiresAt)).map(([name, grant]) => ({
		id: name,
		kind: "capability",
		title: name,
		requestedBy: null,
		granted: true,
		detail: reachOf(grant),
		...(grant.tools ?? []).length ? { tools: grant.tools } : {},
		...(grant.autoConfirm ?? []).length ? { autoConfirm: grant.autoConfirm } : {},
		...grant.expiresAt ? { expiresAt: grant.expiresAt } : {},
		...(grant.agents ?? []).length ? { agents: grant.agents } : {}
	}));
	return {
		pending: [
			...(actions?.pending ?? []).map((action) => ({
				id: action.id,
				kind: "action",
				title: `${action.connector}:${action.tool}`,
				requestedBy: action.requestedBy ?? null,
				detail: "awaiting confirm"
			})),
			...requested,
			...widgets,
			...granted
		],
		onDecide: (item, decision, options) => {
			if (item.kind === "action") actions?.resolve(item.id, decision === "approve" ? "confirm" : "deny");
			else if (item.kind === "capability") resolveCapability(item.id, decision === "approve" ? "granted" : "revoked", options);
			else resolveWidget(item.id, toWidgetApprovalDecision(decision));
		}
	};
}
function mapApprovals(widget, source) {
	const pending = source?.pending.filter((item) => isRecord$4(item) && item.id) ?? [];
	const limitProp = toFiniteNumber(widgetProps(widget).limit);
	const limit = limitProp && limitProp > 0 ? Math.trunc(limitProp) : DEFAULT_LIMIT;
	return {
		items: pending.slice(0, limit),
		total: pending.length
	};
}
//#endregion
//#region ../host/dist/index.js
const PROMPT_RATE_WINDOW_MS = 6e4;
const PROMPT_RATE_MAX = 10;
const promptRateStates = /* @__PURE__ */ new Map();
function getPromptRateState(widgetKey) {
	let state = promptRateStates.get(widgetKey);
	if (!state) {
		state = {
			timestamps: [],
			inFlight: false
		};
		promptRateStates.set(widgetKey, state);
	}
	return state;
}
/**
* The single confirm + rate-limit gate for dispatching a prompt to chat.send.
* Both the sandboxed custom-widget bridge (`handleSendPrompt`) and the trusted
* `builtin:action-form` widget route through THIS function, so there is exactly
* one dispatch privilege: the rate budget (1 in-flight, 10/min, keyed by
* `widgetKey`) and the per-invocation operator confirm are shared, never
* reimplemented. The gate order is fixed: rate check → confirm → send.
*/
async function dispatchRateLimitedPrompt(params) {
	const now = params.now ?? (() => Date.now());
	const rateState = getPromptRateState(params.widgetKey);
	const cutoff = now() - PROMPT_RATE_WINDOW_MS;
	rateState.timestamps = rateState.timestamps.filter((ts) => ts > cutoff);
	if (rateState.inFlight || rateState.timestamps.length >= PROMPT_RATE_MAX) return "rate_limited";
	rateState.inFlight = true;
	try {
		if (!await params.confirmPrompt(params.text)) return "declined";
		rateState.timestamps.push(now());
		await params.sendPrompt(params.text);
		return "sent";
	} finally {
		rateState.inFlight = false;
	}
}
const RPC_METHOD_ALLOWLIST_SET = /* @__PURE__ */ new Set([
	"health",
	"system-presence",
	"usage.status",
	"usage.cost",
	"agents.list",
	"sessions.list",
	"sessions.resolve",
	"sessions.get",
	"sessions.usage",
	"sessions.usage.timeseries",
	"sessions.usage.logs",
	"node.list",
	"node.describe",
	"cron.get",
	"cron.list",
	"cron.status",
	"cron.runs"
]);
/** True when an rpc binding method is in the allowlist (resolve-time re-check). */
function isRpcMethodAllowed(method) {
	return RPC_METHOD_ALLOWLIST_SET.has(method);
}
const STREAM_EVENT_ALLOWLIST_SET = /* @__PURE__ */ new Set([
	"presence",
	"sessions.changed",
	"boardstate.changed"
]);
/** True when a stream binding's event channel is allowlisted (resolve-time re-check). */
function isStreamEventAllowed(event) {
	return STREAM_EVENT_ALLOWLIST_SET.has(event);
}
const DEFAULT_GET_DATA_TIMEOUT_MS = 1e4;
/**
* Pub/sub caps. The payload cap bounds a single broadcast; the rate limiter mirrors
* the sendPrompt limiter (a rolling-window count keyed by stable widget name so a
* remount cannot reset it), sized higher than sendPrompt because pub/sub is a
* cheap in-memory, same-tab backchannel (a filter driving a chart) rather than an
* agent round-trip. The channel-name cap keeps registry keys bounded.
*/
const BUS_MAX_PAYLOAD_BYTES = 8 * 1024;
const BUS_MAX_CHANNEL_LEN = 256;
const BUS_PUBLISH_RATE_WINDOW_MS = 6e4;
const BUS_PUBLISH_RATE_MAX = 60;
const busRateStates = /* @__PURE__ */ new Map();
function getBusRateState(widgetName) {
	let state = busRateStates.get(widgetName);
	if (!state) {
		state = { timestamps: [] };
		busRateStates.set(widgetName, state);
	}
	return state;
}
/**
* Approximate the serialized byte size of a publish payload for the size cap.
* Returns null when the payload cannot be serialized (e.g. a BigInt), which the
* caller treats as a malformed publish. `undefined` (no payload) serializes to 0.
*/
function approxPayloadBytes(payload) {
	let json;
	try {
		json = JSON.stringify(payload);
	} catch {
		return null;
	}
	if (json === void 0) return 0;
	return typeof TextEncoder !== "undefined" ? new TextEncoder().encode(json).length : json.length;
}
const INBOUND_TYPES = /* @__PURE__ */ new Set([
	"dashboard:ready",
	"dashboard:getData",
	"dashboard:getTheme",
	"dashboard:sendPrompt",
	"dashboard:getState",
	"dashboard:setState",
	"dashboard:publish",
	"dashboard:subscribe",
	"dashboard:unsubscribe"
]);
function isRecord$2(value) {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
/**
* Well-formedness filter: a valid inbound message is an object with `v === 1` and
* a known `type`. Anything else is dropped silently (counted for tests). This runs
* AFTER the host's `event.source === iframe.contentWindow` identity check.
*/
function isWellFormedInbound(data) {
	return isRecord$2(data) && data.v === 1 && typeof data.type === "string" && INBOUND_TYPES.has(data.type);
}
/** Creates the parent-side bridge for one approved custom widget. */
function createWidgetBridge(deps) {
	const now = deps.now ?? (() => Date.now());
	const getDataTimeoutMs = deps.getDataTimeoutMs ?? DEFAULT_GET_DATA_TIMEOUT_MS;
	const declaredBindingIds = new Set(deps.manifest.bindingIds);
	const capabilities = new Set(deps.manifest.capabilities);
	let dropped = 0;
	let disposed = false;
	const rateState = getPromptRateState(deps.manifest.name);
	const busRateState = getBusRateState(deps.manifest.name);
	const busUnsubByChannel = /* @__PURE__ */ new Map();
	const pendingTimers = /* @__PURE__ */ new Set();
	function error(code, message, requestId) {
		deps.post({
			v: 1,
			type: "dashboard:error",
			...requestId !== void 0 ? { requestId } : {},
			code,
			message
		});
	}
	async function handleGetData(requestId, bindingId) {
		if (!declaredBindingIds.has(bindingId)) {
			error("binding_denied", `binding not declared in manifest: ${bindingId}`, requestId);
			return;
		}
		const denied = deps.assertBindingAllowed?.(bindingId);
		if (denied) {
			error(denied, `binding not allowed: ${bindingId}`, requestId);
			return;
		}
		let settled = false;
		const timer = setTimeout(() => {
			if (settled || disposed) return;
			settled = true;
			pendingTimers.delete(timer);
			error("timeout", "binding resolution timed out", requestId);
		}, getDataTimeoutMs);
		pendingTimers.add(timer);
		try {
			const data = await deps.resolveBinding(bindingId);
			if (settled || disposed) return;
			settled = true;
			clearTimeout(timer);
			pendingTimers.delete(timer);
			deps.post({
				v: 1,
				type: "dashboard:data",
				requestId,
				bindingId,
				data
			});
		} catch (err) {
			if (settled || disposed) return;
			settled = true;
			clearTimeout(timer);
			pendingTimers.delete(timer);
			error("resolve_failed", err instanceof Error ? err.message : String(err), requestId);
		}
	}
	function handleGetTheme(requestId) {
		deps.post({
			v: 1,
			type: "dashboard:theme",
			requestId,
			tokens: deps.resolveTheme()
		});
	}
	async function handleSendPrompt(requestId, text) {
		if (!capabilities.has("prompt:send")) {
			error("capability_denied", "widget lacks the prompt:send capability", requestId);
			return;
		}
		try {
			const outcome = await dispatchRateLimitedPrompt({
				widgetKey: deps.manifest.name,
				text,
				confirmPrompt: deps.confirmPrompt,
				sendPrompt: deps.sendPrompt,
				now
			});
			if (disposed) return;
			if (outcome === "rate_limited") error("rate_limited", "prompt send rate limit exceeded", requestId);
			else if (outcome === "declined") error("prompt_declined", "operator declined the prompt", requestId);
		} catch (err) {
			if (!disposed) error("resolve_failed", err instanceof Error ? err.message : String(err), requestId);
		}
	}
	async function handleGetState(requestId) {
		if (!capabilities.has("state:persist") || !deps.getWidgetState) {
			error("capability_denied", "widget lacks the state:persist capability", requestId);
			return;
		}
		try {
			const result = await deps.getWidgetState();
			if (disposed) return;
			deps.post({
				v: 1,
				type: "dashboard:state",
				requestId,
				state: result.state,
				...result.version !== void 0 ? { version: result.version } : {}
			});
		} catch (err) {
			if (!disposed) error("resolve_failed", err instanceof Error ? err.message : String(err), requestId);
		}
	}
	async function handleSetState(requestId, blob) {
		if (!capabilities.has("state:persist") || !deps.setWidgetState) {
			error("capability_denied", "widget lacks the state:persist capability", requestId);
			return;
		}
		try {
			const { version } = await deps.setWidgetState(blob);
			if (disposed) return;
			deps.post({
				v: 1,
				type: "dashboard:state",
				requestId,
				state: blob,
				version
			});
		} catch (err) {
			if (!disposed) error("resolve_failed", err instanceof Error ? err.message : String(err), requestId);
		}
	}
	function handlePublish(channel, payload, requestId) {
		if (!capabilities.has("bus:pubsub")) {
			error("capability_denied", "widget lacks the bus:pubsub capability", requestId);
			return;
		}
		if (!deps.bus) return;
		const size = approxPayloadBytes(payload);
		if (size === null) {
			error("malformed", "publish payload is not serializable", requestId);
			return;
		}
		if (size > BUS_MAX_PAYLOAD_BYTES) {
			error("payload_too_large", `publish payload exceeds ${BUS_MAX_PAYLOAD_BYTES} bytes`, requestId);
			return;
		}
		const cutoff = now() - BUS_PUBLISH_RATE_WINDOW_MS;
		busRateState.timestamps = busRateState.timestamps.filter((ts) => ts > cutoff);
		if (busRateState.timestamps.length >= BUS_PUBLISH_RATE_MAX) {
			error("rate_limited", "publish rate limit exceeded", requestId);
			return;
		}
		busRateState.timestamps.push(now());
		deps.bus.publish(channel, payload);
	}
	function handleSubscribe(channel) {
		if (!capabilities.has("bus:pubsub") || !deps.bus) {
			if (!capabilities.has("bus:pubsub")) error("capability_denied", "widget lacks the bus:pubsub capability");
			return;
		}
		if (busUnsubByChannel.has(channel)) return;
		const unsub = deps.bus.subscribe(channel, (ch, payload) => {
			if (disposed) return;
			deps.post({
				v: 1,
				type: "dashboard:message",
				channel: ch,
				payload
			});
		});
		busUnsubByChannel.set(channel, unsub);
	}
	function handleUnsubscribe(channel) {
		const unsub = busUnsubByChannel.get(channel);
		if (unsub) {
			busUnsubByChannel.delete(channel);
			unsub();
		}
	}
	function handleMessage(data) {
		if (disposed) return false;
		if (!isWellFormedInbound(data)) {
			dropped += 1;
			return false;
		}
		switch (data.type) {
			case "dashboard:ready": return true;
			case "dashboard:getData": {
				const requestId = typeof data.requestId === "string" ? data.requestId : null;
				const bindingId = typeof data.bindingId === "string" ? data.bindingId : null;
				if (requestId === null || bindingId === null) {
					dropped += 1;
					return false;
				}
				handleGetData(requestId, bindingId);
				return true;
			}
			case "dashboard:getTheme": {
				const requestId = typeof data.requestId === "string" ? data.requestId : null;
				if (requestId === null) {
					dropped += 1;
					return false;
				}
				handleGetTheme(requestId);
				return true;
			}
			case "dashboard:sendPrompt": {
				const requestId = typeof data.requestId === "string" ? data.requestId : null;
				const text = typeof data.text === "string" ? data.text : null;
				if (requestId === null || text === null || !text.trim()) {
					dropped += 1;
					return false;
				}
				handleSendPrompt(requestId, text);
				return true;
			}
			case "dashboard:getState": {
				const requestId = typeof data.requestId === "string" ? data.requestId : null;
				if (requestId === null) {
					dropped += 1;
					return false;
				}
				handleGetState(requestId);
				return true;
			}
			case "dashboard:setState": {
				const requestId = typeof data.requestId === "string" ? data.requestId : null;
				if (requestId === null || !Object.hasOwn(data, "state")) {
					dropped += 1;
					return false;
				}
				handleSetState(requestId, data.state);
				return true;
			}
			case "dashboard:publish": {
				const channel = typeof data.channel === "string" ? data.channel : null;
				const requestId = typeof data.requestId === "string" ? data.requestId : void 0;
				if (channel === null || !channel.trim() || channel.length > BUS_MAX_CHANNEL_LEN) {
					dropped += 1;
					return false;
				}
				if (!("payload" in data)) {
					dropped += 1;
					return false;
				}
				handlePublish(channel, data.payload, requestId);
				return true;
			}
			case "dashboard:subscribe": {
				const channel = typeof data.channel === "string" ? data.channel : null;
				if (channel === null || !channel.trim() || channel.length > BUS_MAX_CHANNEL_LEN) {
					dropped += 1;
					return false;
				}
				handleSubscribe(channel);
				return true;
			}
			case "dashboard:unsubscribe": {
				const channel = typeof data.channel === "string" ? data.channel : null;
				if (channel === null || !channel.trim() || channel.length > BUS_MAX_CHANNEL_LEN) {
					dropped += 1;
					return false;
				}
				handleUnsubscribe(channel);
				return true;
			}
			default:
				dropped += 1;
				return false;
		}
	}
	async function push(bindingId) {
		if (disposed || !declaredBindingIds.has(bindingId) || deps.assertBindingAllowed?.(bindingId)) return;
		try {
			const data = await deps.resolveBinding(bindingId);
			if (!disposed) deps.post({
				v: 1,
				type: "dashboard:push",
				bindingId,
				data
			});
		} catch {}
	}
	return {
		handleMessage,
		push,
		get droppedCount() {
			return dropped;
		},
		dispose() {
			disposed = true;
			for (const timer of pendingTimers) clearTimeout(timer);
			pendingTimers.clear();
			for (const unsub of busUnsubByChannel.values()) unsub();
			busUnsubByChannel.clear();
			rateState.inFlight = false;
		}
	};
}
const dashboardPresence = /* @__PURE__ */ new WeakMap();
function getPresenceState(host) {
	let state = dashboardPresence.get(host);
	if (!state) {
		state = {
			entries: /* @__PURE__ */ new Map(),
			self: null,
			pendingSelfSlug: null
		};
		dashboardPresence.set(host, state);
	}
	return state;
}
function prunePresence(state, now) {
	for (const [operator, entry] of state.entries) if (entry.at + 3e4 <= now) state.entries.delete(operator);
}
/** Operator ids (excluding self, excluding stale) viewing `tabSlug`, freshest first. */
function presenceForTab(host, tabSlug, now = Date.now()) {
	const state = dashboardPresence.get(host);
	if (!state) return [];
	prunePresence(state, now);
	return [...state.entries.entries()].filter(([operator, entry]) => entry.tabSlug === tabSlug && operator !== state.self).toSorted((a, b) => b[1].at - a[1].at).map(([operator]) => operator);
}
/** Drop all presence for a host (full teardown). */
function clearPresence(host) {
	dashboardPresence.delete(host);
}
/**
* Announce that this client is viewing `tabSlug`. Fire-and-forget: a failed
* heartbeat is not a product error. The identity in the resulting broadcast is
* resolved server-side (the payload carries only the tab slug).
*/
function pingPresence(host, transport, tabSlug) {
	if (!transport) return;
	const state = getPresenceState(host);
	if (state.self === null) state.pendingSelfSlug = tabSlug;
	transport.request("dashboard.presence.ping", { tabSlug }).catch(() => {});
}
/** Broadcast event the host emits on any workspace mutation (SPEC §5). */
const CHANGED_EVENT = "boardstate.changed";
const dashboardStates = /* @__PURE__ */ new WeakMap();
const dashboardEventUnsubscribers = /* @__PURE__ */ new WeakMap();
const dashboardEventTransports = /* @__PURE__ */ new WeakMap();
const dashboardPollTimers = /* @__PURE__ */ new WeakMap();
const dashboardPollActive = /* @__PURE__ */ new WeakMap();
/** Default data-refresh interval (ms); the 30–60s window, floored at 10s. */
const DASHBOARD_POLL_INTERVAL_MS = 45e3;
const dashboardActiveDragCancel = /* @__PURE__ */ new WeakMap();
/**
* Register the teardown for an active drag on `host`. The view calls this when a
* drag begins; `cancel` must remove its window listeners and make any later
* pointerup a no-op. A previously registered drag is cancelled first so only one
* drag is ever live per host.
*/
function registerActiveDrag(host, cancel) {
	dashboardActiveDragCancel.get(host)?.();
	dashboardActiveDragCancel.set(host, cancel);
}
/** Clear the active-drag teardown for `host` once the drag settles normally. */
function clearActiveDrag(host) {
	dashboardActiveDragCancel.delete(host);
}
/** Cancel any in-flight drag on `host` (used by stopDashboard and re-registration). */
function cancelActiveDrag(host) {
	const cancel = dashboardActiveDragCancel.get(host);
	if (cancel) {
		dashboardActiveDragCancel.delete(host);
		cancel();
	}
}
function getDashboardState(host) {
	let state = dashboardStates.get(host);
	if (!state) {
		state = {
			loading: false,
			loaded: false,
			error: null,
			workspace: null,
			activeSlug: null,
			hiddenMenuOpen: false,
			pendingWidgetIds: /* @__PURE__ */ new Set(),
			actionError: null,
			requestUpdate: null
		};
		dashboardStates.set(host, state);
	}
	return state;
}
function notify(state) {
	state.requestUpdate?.();
}
function isRecord$1(value) {
	return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
function readNumber(value, fallback = 0) {
	return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}
function formatError(error) {
	if (error instanceof Error && error.message.trim()) return error.message.trim();
	if (typeof error === "string" && error.trim()) return error.trim();
	return "Unknown dashboard error.";
}
/** Load the workspace document; seeds `activeSlug` from the requested deep-link slug. */
async function loadWorkspace(state, transport, opts) {
	if (!transport) return;
	if (!opts?.silent) {
		state.loading = true;
		state.error = null;
		notify(state);
	}
	try {
		const payload = await transport.request("dashboard.workspace.get", {});
		const workspace = normalizeWorkspace(isRecord$1(payload) && "doc" in payload ? payload.doc : payload);
		state.workspace = workspace;
		state.activeSlug = resolveActiveSlug(workspace, opts?.requestedSlug ?? state.activeSlug);
		state.error = null;
		state.loaded = true;
	} catch (err) {
		state.error = formatError(err);
	} finally {
		state.loading = false;
		notify(state);
	}
}
/**
* Subscribe to `boardstate.changed` and refetch on a newer version (skips
* stale/own-echo events by comparing `workspaceVersion`). The transport delivers
* the event payload directly.
*/
function subscribeToDashboardEvents(host, state, transport) {
	if (!transport) {
		stopDashboardEvents(host);
		return;
	}
	if (dashboardEventTransports.get(host) === transport) return;
	stopDashboardEvents(host);
	const unsubscribe = transport.addEventListener(CHANGED_EVENT, (raw) => {
		const incomingVersion = readNumber((isRecord$1(raw) ? raw : void 0)?.workspaceVersion, NaN);
		const currentVersion = state.workspace?.workspaceVersion ?? -1;
		if (Number.isFinite(incomingVersion) && incomingVersion <= currentVersion) return;
		loadWorkspace(state, transport, { silent: true });
	});
	dashboardEventUnsubscribers.set(host, unsubscribe);
	dashboardEventTransports.set(host, transport);
}
function stopDashboardEvents(host) {
	dashboardEventUnsubscribers.get(host)?.();
	dashboardEventUnsubscribers.delete(host);
	dashboardEventTransports.delete(host);
}
/**
* Start (idempotently) the per-host data-refresh timer. The timer fires `onTick`
* every `intervalMs`, but ONLY while the document is visible — a background tab
* skips the tick so we don't hammer the gateway when nobody's watching. Passing a
* null transport stops any running timer (disconnect). A second call with a live
* transport is a no-op so re-renders don't stack timers.
*/
function startBindingPolling(host, transport, onTick, intervalMs = DASHBOARD_POLL_INTERVAL_MS) {
	if (!transport) {
		stopBindingPolling(host);
		return;
	}
	if (dashboardPollActive.get(host)) return;
	const timer = setInterval(() => {
		if (typeof document !== "undefined" && document.visibilityState === "hidden") return;
		onTick();
	}, Math.max(1e4, intervalMs));
	dashboardPollTimers.set(host, timer);
	dashboardPollActive.set(host, true);
}
/** Stop the per-host data-refresh timer (tab-leave/disconnect). */
function stopBindingPolling(host) {
	const timer = dashboardPollTimers.get(host);
	if (timer !== void 0) {
		clearInterval(timer);
		dashboardPollTimers.delete(host);
	}
	dashboardPollActive.delete(host);
}
/** Full lifecycle teardown for the client's `stop` hook. */
function stopDashboard(host) {
	cancelActiveDrag(host);
	stopDashboardEvents(host);
	stopBindingPolling(host);
	clearPresence(host);
}
function replaceWidget(workspace, slug, widgetId, update) {
	return {
		...workspace,
		tabs: workspace.tabs.map((tab) => tab.slug !== slug ? tab : {
			...tab,
			widgets: tab.widgets.map((widget) => widget.id === widgetId ? update(widget) : widget)
		})
	};
}
function removeWidget(workspace, slug, widgetId) {
	return {
		...workspace,
		tabs: workspace.tabs.map((tab) => tab.slug !== slug ? tab : {
			...tab,
			widgets: tab.widgets.filter((widget) => widget.id !== widgetId)
		})
	};
}
/**
* Run an optimistic mutation: apply `optimistic` locally, fire the RPC, and revert
* to the pre-mutation snapshot on failure (surfacing `actionError` for a toast).
* All shell mutations funnel through here so revert semantics stay consistent.
*/
async function optimisticMutation(state, transport, params) {
	if (!transport || !state.workspace) return;
	const previous = state.workspace;
	const optimistic = params.optimistic(previous);
	state.workspace = optimistic;
	state.pendingWidgetIds.add(params.widgetId);
	state.actionError = null;
	notify(state);
	try {
		await transport.request(params.method, params.rpcParams);
	} catch (err) {
		if (state.workspace === optimistic) state.workspace = previous;
		state.actionError = formatError(err);
	} finally {
		state.pendingWidgetIds.delete(params.widgetId);
		notify(state);
	}
}
function moveWidget(state, transport, params) {
	return optimisticMutation(state, transport, {
		widgetId: params.widgetId,
		method: "dashboard.widget.move",
		rpcParams: {
			tab: params.slug,
			id: params.widgetId,
			grid: params.grid
		},
		optimistic: (workspace) => replaceWidget(workspace, params.slug, params.widgetId, (widget) => ({
			...widget,
			grid: params.grid
		}))
	});
}
function setWidgetCollapsed(state, transport, params) {
	return optimisticMutation(state, transport, {
		widgetId: params.widgetId,
		method: "dashboard.widget.update",
		rpcParams: {
			tab: params.slug,
			id: params.widgetId,
			patch: { collapsed: params.collapsed }
		},
		optimistic: (workspace) => replaceWidget(workspace, params.slug, params.widgetId, (widget) => ({
			...widget,
			collapsed: params.collapsed
		}))
	});
}
function updateWidgetTitle(state, transport, params) {
	return optimisticMutation(state, transport, {
		widgetId: params.widgetId,
		method: "dashboard.widget.update",
		rpcParams: {
			tab: params.slug,
			id: params.widgetId,
			patch: { title: params.title }
		},
		optimistic: (workspace) => replaceWidget(workspace, params.slug, params.widgetId, (widget) => ({
			...widget,
			title: params.title
		}))
	});
}
/**
* Pin a temporary (ephemeral) Living Answer: clear its `ephemeral` flag so the
* store's TTL sweep never removes it. Mirrors the other widget.update actions —
* `ephemeral: null` is the clear signal the store's patch reader understands.
*/
function pinWidget(state, transport, params) {
	return optimisticMutation(state, transport, {
		widgetId: params.widgetId,
		method: "dashboard.widget.update",
		rpcParams: {
			tab: params.slug,
			id: params.widgetId,
			patch: { ephemeral: null }
		},
		optimistic: (workspace) => replaceWidget(workspace, params.slug, params.widgetId, (widget) => {
			const { ephemeral: _ephemeral, ...rest } = widget;
			return rest;
		})
	});
}
function hideWidget(state, transport, params) {
	return optimisticMutation(state, transport, {
		widgetId: params.widgetId,
		method: "dashboard.widget.update",
		rpcParams: {
			tab: params.slug,
			id: params.widgetId,
			patch: { hidden: true }
		},
		optimistic: (workspace) => removeWidget(workspace, params.slug, params.widgetId)
	});
}
function removeWidgetFromTab(state, transport, params) {
	return optimisticMutation(state, transport, {
		widgetId: params.widgetId,
		method: "dashboard.widget.remove",
		rpcParams: {
			tab: params.slug,
			id: params.widgetId
		},
		optimistic: (workspace) => removeWidget(workspace, params.slug, params.widgetId)
	});
}
function moveWidgetToTab(state, transport, params) {
	return optimisticMutation(state, transport, {
		widgetId: params.widgetId,
		method: "dashboard.widget.move",
		rpcParams: {
			tab: params.fromSlug,
			id: params.widgetId,
			toTab: params.toSlug
		},
		optimistic: (workspace) => {
			const widget = workspace.tabs.find((tab) => tab.slug === params.fromSlug)?.widgets.find((w) => w.id === params.widgetId);
			if (!widget) return workspace;
			return {
				...workspace,
				tabs: workspace.tabs.map((tab) => {
					if (tab.slug === params.fromSlug) return {
						...tab,
						widgets: tab.widgets.filter((w) => w.id !== params.widgetId)
					};
					if (tab.slug === params.toSlug) return {
						...tab,
						widgets: [...tab.widgets, widget]
					};
					return tab;
				})
			};
		}
	});
}
/**
* Set a tab's content layout ("grid" | "full") → `dashboard.tab.update` (WRITE).
* Optimistically flips the tab layout so the full-bleed toggle feels instant, then
* reverts to the pre-mutation snapshot on failure (surfacing `actionError`). The
* revert is guarded like the widget mutations so a concurrent refetch isn't stomped.
*/
async function setTabLayout(state, transport, params) {
	if (!transport || !state.workspace) return;
	const previous = state.workspace;
	const optimistic = {
		...previous,
		tabs: previous.tabs.map((tab) => tab.slug === params.slug ? {
			...tab,
			layout: params.layout
		} : tab)
	};
	state.workspace = optimistic;
	state.actionError = null;
	notify(state);
	try {
		await transport.request("dashboard.tab.update", {
			slug: params.slug,
			patch: { layout: params.layout }
		});
	} catch (err) {
		if (state.workspace === optimistic) state.workspace = previous;
		state.actionError = formatError(err);
		notify(state);
	}
}
/**
* Restore the most recent workspace snapshot via the EXISTING undo write path
* (time-travel reuses `dashboard.workspace.undo`; no new write RPC). The resulting
* `boardstate.changed` broadcast refetches, but we also reload eagerly so the caller
* sees the reverted doc without waiting for the echo. A failure surfaces `actionError`.
*/
async function undoWorkspace(state, transport) {
	if (!transport) return;
	state.actionError = null;
	notify(state);
	try {
		await transport.request("dashboard.workspace.undo", {});
		await loadWorkspace(state, transport, { silent: true });
	} catch (err) {
		state.actionError = formatError(err);
		notify(state);
	}
}
/**
* Approve or reject a pending custom widget (operator-only) → `dashboard.widget.approve`
* (WRITE). The registry is not part of the optimistic widget model, so this fires
* the RPC and lets the resulting `boardstate.changed` broadcast refetch the new
* status; a failure surfaces `actionError` for the toast.
*/
async function approveWidget(state, transport, params) {
	if (!transport) return;
	state.actionError = null;
	notify(state);
	try {
		await transport.request("dashboard.widget.approve", {
			name: params.name,
			decision: params.decision
		});
	} catch (err) {
		state.actionError = formatError(err);
		notify(state);
	}
}
/**
* Operator grant/revoke of a connector's data/tool capability (SPEC §17). A partial
* grant (§17.1) passes the SUBSET of `connector:tool` ids the operator ticked; omitted
* ⇒ approve-all (the full requested set).
*/
async function approveCapability(state, transport, params) {
	if (!transport) return;
	state.actionError = null;
	notify(state);
	try {
		await transport.request("dashboard.capability.approve", {
			name: params.name,
			decision: params.decision,
			...params.tools !== void 0 ? { tools: params.tools } : {},
			...params.autoConfirm !== void 0 ? { autoConfirm: params.autoConfirm } : {},
			...params.expiresAt !== void 0 ? { expiresAt: params.expiresAt } : {}
		});
	} catch (err) {
		state.actionError = formatError(err);
		notify(state);
	}
}
/**
* Fetch the strict workspace doc and serialize it (optionally a chosen tab subset)
* for download. Reads the canonical `workspace.json` from the gateway so the export
* round-trips through the write-time validator on re-import (the UI read model is
* lossy). Throws when disconnected; the caller surfaces the failure.
*/
async function exportWorkspace(transport, options = {}) {
	if (!transport) throw new Error("Not connected.");
	const doc = workspaceDocFromPayload(await transport.request("dashboard.workspace.get", {}));
	return {
		filename: workspaceExportFilename(),
		json: serializeWorkspaceExport(doc, options)
	};
}
/**
* Import a workspace JSON file: parse, coerce every custom widget to `pending` (so
* the approval gate runs — an import NEVER auto-approves), then apply via the
* existing `dashboard.workspace.replace`, which RE-VALIDATES the doc server-side.
* A parse or validation failure surfaces as an `actionError` toast; returns whether
* the import applied.
*/
async function importWorkspace(state, transport, text) {
	if (!transport) return false;
	state.actionError = null;
	notify(state);
	try {
		const doc = sanitizeImportedWorkspace(parseWorkspaceImport(text));
		await transport.request("dashboard.workspace.replace", { doc });
		await loadWorkspace(state, transport, { silent: true });
		return true;
	} catch (err) {
		state.actionError = formatError(err);
		notify(state);
		return false;
	}
}
/**
* Install a template recipe (issue #60): INSTALL = IMPORT. The recipe's board is merged
* with the grants its `grantsManifest` declares (each `requested`) and run through the
* SAME `sanitizeImportedWorkspace` re-pend as any imported workspace
* (`buildRecipeImportDoc`), then applied via `dashboard.workspace.replace` (which the
* store re-validates AND re-pends via `reconcileReplaceApproval`). So installing can
* never grant: every manifest grant lands `requested`, every custom widget `pending`.
* A validation failure surfaces as an `actionError` toast; returns whether it applied.
*/
async function installRecipe(state, transport, recipe) {
	if (!transport) return false;
	state.actionError = null;
	notify(state);
	try {
		const doc = buildRecipeImportDoc(recipe);
		await transport.request("dashboard.workspace.replace", { doc });
		await loadWorkspace(state, transport, { silent: true });
		return true;
	} catch (err) {
		state.actionError = formatError(err);
		notify(state);
		return false;
	}
}
/**
* Resolve a widget binding into a value the builtin renderers consume. Wire is:
* - `static`: literal value from the binding.
* - `rpc`: resolved CLIENT-SIDE on the page's own transport.
* - `file`: served by `dashboard.data.read`; the JSON pointer is applied server-side.
* - `stream`/`computed`: never a one-shot read (see `subscribeToStreamBinding` /
*   `resolveComputedBinding`); guarded so a stream binding can never be mistaken for
*   a `file` read against an empty path.
* - `mcp`: resolved host-side through the connector broker's readOnly action path
*   (`resolveMcpBinding` → `dashboard.action.invoke`); NEVER routed to `data.read`.
*
* `dashboard.data.read` serves file/static only and answers rpc bindings with
* `{ code: "binding_client_resolved" }`, so rpc never routes through it.
*/
async function resolveBinding(transport, binding) {
	try {
		if (binding.source === "static") return { value: binding.value };
		if (!transport) return { error: "Not connected." };
		if (binding.source === "rpc") {
			if (!binding.method) return { error: "Binding is missing an rpc method." };
			return { value: applyPointer(await transport.request(binding.method, binding.params ?? {}), binding.pointer) };
		}
		if (binding.source === "stream") return { error: "Stream bindings resolve via subscription, not a one-shot read." };
		if (binding.source === "computed") return { error: "Computed bindings resolve from sibling values, not a one-shot read." };
		if (binding.source === "mcp") return await resolveMcpBinding(transport, binding);
		const payload = await transport.request("dashboard.data.read", { binding });
		return { value: isRecord$1(payload) && "data" in payload ? payload.data : payload };
	} catch (err) {
		return { error: formatError(err) };
	}
}
/**
* Resolve an `mcp` read binding (SPEC §18 / #45) through the connector broker via
* the PURE-READ verb `dashboard.connector.read`, which AND-gates the tool (granted +
* connector-configured + manifest-hash unchanged) and executes a `readOnly` granted
* tool DIRECTLY, returning its result.
*
* readOnly-ONLY, fail-safe (epic invariant #5): a binding may only READ. `connector.read`
* REFUSES a non-readOnly tool outright — it never parks a pending action. This matters
* because a read binding re-resolves on every refresh: routing through `action.invoke`
* would have PARKED a pending mutation into the operator queue on each refresh (queue
* spam, and an operator confirm would then fire the mutation). An ungranted / re-pended
* tool surfaces the engine's `capability_pending` through the standard binding-error card.
*/
async function resolveMcpBinding(transport, binding) {
	if (!binding.connector || !binding.tool) return { error: "mcp binding is missing a connector or tool." };
	return { value: applyPointer(mcpReadValue(await transport.request("dashboard.connector.read", {
		connector: binding.connector,
		tool: binding.tool,
		...binding.args ? { args: binding.args } : {}
	})), binding.pointer) };
}
/**
* Extract the value a readOnly tool returns from `dashboard.action.invoke`'s
* `{ content, structuredContent? }` result. Prefer the parsed `structuredContent`
* (the machine-shaped payload) when present, else the raw `content`, else the whole
* result — so a widget binds the useful data, not the MCP envelope.
*/
function mcpReadValue(result) {
	if (isRecord$1(result)) {
		if ("structuredContent" in result && result.structuredContent !== void 0) return result.structuredContent;
		if ("content" in result) return result.content;
	}
	return result;
}
/** Recursively collect finite numbers from a value (numbers + nested arrays). */
function collectNumbers(value, out) {
	if (typeof value === "number" && Number.isFinite(value)) out.push(value);
	else if (Array.isArray(value)) for (const entry of value) collectNumbers(entry, out);
}
/** Count elements across an input value: an array contributes its length, a defined scalar 1. */
function countElements(value) {
	if (Array.isArray(value)) return value.length;
	return value === void 0 || value === null ? 0 : 1;
}
/** Interpolate `{i}` placeholders in `template` with the i-th input value; no eval. */
function formatComputed(template, inputValues) {
	return template.replace(/\{(\d+)\}/g, (_match, digits) => {
		const value = inputValues[Number(digits)];
		if (typeof value === "string") return value;
		if (typeof value === "number" || typeof value === "boolean" || typeof value === "bigint") return String(value);
		return value === void 0 || value === null ? "" : JSON.stringify(value) ?? "";
	});
}
/**
* Resolve a `computed` binding CLIENT-SIDE from its already-resolved sibling input
* values via a FIXED whitelisted op — a switch, never an expression language or
* eval. `inputValues` are the resolved values of the binding's `inputs`, in order.
* - `sum|avg|min|max`: reduce the finite numbers flattened out of the inputs
*   (empty → `0` for sum, `null` for avg/min/max).
* - `count`: total element count across the inputs (array → length).
* - `last`: the last input's raw value.
* - `pick`: the JSON pointer `arg` applied to the FIRST input.
* - `format`: the template `arg` with `{i}` placeholders filled from the inputs.
*/
function resolveComputedBinding(op, inputValues, arg) {
	switch (op) {
		case "sum":
		case "avg":
		case "min":
		case "max": {
			const nums = [];
			for (const value of inputValues) collectNumbers(value, nums);
			if (op === "sum") return { value: nums.reduce((total, n) => total + n, 0) };
			if (nums.length === 0) return { value: null };
			if (op === "avg") return { value: nums.reduce((total, n) => total + n, 0) / nums.length };
			return { value: op === "min" ? Math.min(...nums) : Math.max(...nums) };
		}
		case "count": return { value: inputValues.reduce((total, v) => total + countElements(v), 0) };
		case "last": return { value: inputValues.length ? inputValues[inputValues.length - 1] : null };
		case "pick": return { value: applyPointer(inputValues[0], arg) };
		case "format": return { value: formatComputed(arg ?? "", inputValues) };
		default: return { error: `Unknown computed op: ${op}` };
	}
}
/**
* Subscribe a `stream` binding to its allowlisted broadcast channel. Each event
* payload pushes `applyPointer(payload, pointer)` to `onValue`; the returned fn
* unsubscribes (call on unmount). This NEVER opens a socket — it multiplexes over
* the transport's existing event stream via `addEventListener`. A missing transport
* or a non-allowlisted event id subscribes to nothing (defense-in-depth over the
* write-time schema gate), so a stream binding can never listen on an arbitrary
* channel.
*/
function subscribeToStreamBinding(transport, binding, onValue) {
	const event = binding.event;
	if (!transport || !event || !isStreamEventAllowed(event)) return () => {};
	return transport.addEventListener(event, (payload) => {
		try {
			onValue({ value: applyPointer(payload, binding.pointer) });
		} catch (err) {
			onValue({ error: formatError(err) });
		}
	});
}
const WIDGET_THEME_TOKENS = [
	"--bg",
	"--card",
	"--card-foreground",
	"--text",
	"--muted",
	"--border",
	"--accent",
	"--accent-foreground",
	"--radius",
	"--radius-sm",
	"--font-sans",
	"--font-mono"
];
/** Read the standard widget theme tokens from the document root's computed styles. */
function readThemeTokensFromRoot() {
	const tokens = {};
	if (typeof document === "undefined" || typeof getComputedStyle !== "function") return tokens;
	const styles = getComputedStyle(document.documentElement);
	for (const token of WIDGET_THEME_TOKENS) {
		const value = styles.getPropertyValue(token).trim();
		if (value) tokens[token] = value;
	}
	return tokens;
}
/**
* Build a `widget.id`-bound state accessor over a transport. `dashboard.widget.state.get`
* answers `{ state, version? }`; `dashboard.widget.state.set` answers `{ version }`.
* The widgetId is closed over here (host-tracked), never read from a child message.
*/
function createWidgetStateAccessor(transport, widgetId) {
	return {
		get: async () => {
			const record = await transport.request("dashboard.widget.state.get", { widgetId });
			return {
				state: record?.state ?? null,
				...typeof record?.version === "number" ? { version: record.version } : {}
			};
		},
		set: async (blob) => {
			const version = (await transport.request("dashboard.widget.state.set", {
				widgetId,
				state: blob
			}))?.version;
			return { version: typeof version === "number" ? version : 0 };
		}
	};
}
/**
* Mount the parent bridge on `iframe`: post to the child with targetOrigin "*"
* (opaque origin), install the window `message` listener whose IDENTITY accept
* filter (`event.source === iframe.contentWindow`) runs BEFORE the bridge sees any
* message, and wire the injected bridge deps. Returns a dispose fn that removes the
* listener and disposes the bridge.
*/
function mountCustomWidget(iframe, options) {
	const post = (message) => {
		iframe.contentWindow?.postMessage(message, "*");
	};
	const bridge = createWidgetBridge({
		...options,
		post
	});
	const onMessage = (event) => {
		if (event.source !== iframe.contentWindow) return;
		bridge.handleMessage(event.data);
	};
	const view = iframe.ownerDocument?.defaultView ?? (typeof window !== "undefined" ? window : null);
	view?.addEventListener("message", onMessage);
	return () => {
		view?.removeEventListener("message", onMessage);
		bridge.dispose();
	};
}
/**
* Capabilities accepted from a widget manifest. Superset of core's
* `DashboardWidgetCapability` — `bus:pubsub` (pub/sub) is a real, bridge-gated
* capability the core enum does not yet enumerate (see host manifest note). Kept as
* a string allowlist so an unknown capability is dropped, then narrowed to the core
* type at the boundary.
*/
const ACCEPTED_CAPABILITIES = /* @__PURE__ */ new Set([
	"data:read",
	"prompt:send",
	"state:persist",
	"bus:pubsub"
]);
/** Builds the served asset URL for a widget file under the widget-asset route. */
function widgetAssetUrl(basePath, name, file) {
	return `${basePath.replace(/\/+$/, "")}/widgets/${encodeURIComponent(name)}/${file.split("/").map((segment) => encodeURIComponent(segment)).join("/")}`;
}
/** Fetches and shapes a widget's manifest into the bridge's read model. */
async function loadWidgetManifestView(basePath, name) {
	if (typeof fetch !== "function") return null;
	try {
		const res = await fetch(widgetAssetUrl(basePath, name, "widget.json"), {
			method: "GET",
			credentials: "same-origin",
			headers: { Accept: "application/json" }
		});
		if (!res.ok) return null;
		const parsed = await res.json();
		if (typeof parsed !== "object" || parsed === null) return null;
		const record = parsed;
		return {
			name,
			bindingIds: (Array.isArray(record.bindings) ? record.bindings : []).map((binding) => typeof binding === "object" && binding !== null ? binding.id : void 0).filter((id) => typeof id === "string"),
			capabilities: (Array.isArray(record.capabilities) ? record.capabilities : []).filter((cap) => typeof cap === "string" && ACCEPTED_CAPABILITIES.has(cap))
		};
	} catch {
		return null;
	}
}
function isRecord$3(value) {
	return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}
function count(value) {
	return typeof value === "number" && Number.isFinite(value) ? value : 0;
}
/**
* Re-hydrate the per-entry change summary the store computed (SPEC: history list
* rows). It crosses the wire as plain JSON, so re-normalize it defensively here —
* a pre-summary host (or a malformed entry) simply yields no summary and the row
* falls back to version + time. Kept in lock-step with `DashboardHistorySummary`.
*/
function normalizeSummary(value) {
	if (!isRecord$3(value)) return;
	return {
		added: count(value.added),
		removed: count(value.removed),
		moved: count(value.moved),
		retitled: count(value.retitled),
		tabsChanged: count(value.tabsChanged),
		total: count(value.total)
	};
}
/** Fetch the ring metadata (newest-first) via the read-only history.list RPC. */
async function loadHistoryList(transport) {
	if (!transport) return [];
	const payload = await transport.request("dashboard.workspace.history.list", {});
	return (isRecord$3(payload) && Array.isArray(payload.entries) ? payload.entries : []).filter(isRecord$3).map((entry) => {
		const summary = normalizeSummary(entry.summary);
		return {
			version: typeof entry.version === "number" ? entry.version : 0,
			savedAt: typeof entry.savedAt === "string" ? entry.savedAt : "",
			bytes: typeof entry.bytes === "number" ? entry.bytes : 0,
			...summary ? { summary } : {}
		};
	}).filter((entry) => entry.version > 0);
}
/** Fetch one full snapshot doc via the read-only history.get RPC. */
async function loadHistorySnapshot(transport, version) {
	if (!transport) return null;
	const payload = await transport.request("dashboard.workspace.history.get", { version });
	return normalizeWorkspace(isRecord$3(payload) && "doc" in payload ? payload.doc : payload);
}
async function fetchTextCapped(url, maxBytes, label) {
	if (typeof fetch !== "function") throw new Error("This browser cannot fetch the widget gallery.");
	const res = await fetch(url, {
		method: "GET",
		credentials: "omit",
		headers: { Accept: "application/json" }
	});
	if (!res.ok) throw new Error(`${label} request failed (${res.status}).`);
	const text = await res.text();
	if (galleryByteLength(text) > maxBytes) throw new Error(`${label} is too large (max ${Math.floor(maxBytes / 1024)} KB).`);
	return text;
}
/**
* Fetch and parse a registry `index.json` (CLIENT fetch). Relative `manifestUrl`s
* resolve against the index URL; malformed entries are dropped rather than throwing.
*/
async function fetchGalleryIndex(indexUrl) {
	return parseGalleryIndex(await fetchTextCapped(indexUrl, GALLERY_INDEX_MAX_BYTES, "The gallery index"), indexUrl);
}
/**
* Fetch a widget bundle (CLIENT fetch) and shape-check it. Enforces the 512 KB cap
* before parsing. Authoritative manifest validation happens server-side on install.
*/
async function fetchWidgetBundle(bundleUrl) {
	return parseWidgetBundle(await fetchTextCapped(bundleUrl, GALLERY_BUNDLE_MAX_BYTES, "The widget bundle"));
}
/**
* Fetch the `recipes` half of a registry `index.json` (CLIENT fetch), sibling of the
* widget entries. An index with no `recipes` key yields `[]`; malformed entries drop.
*/
async function fetchGalleryRecipes(indexUrl) {
	return parseRecipeIndex(await fetchTextCapped(indexUrl, GALLERY_INDEX_MAX_BYTES, "The gallery index"), indexUrl);
}
/**
* Fetch a recipe bundle (CLIENT fetch) and fully validate it with the shared
* `validateRecipe`. Enforces the 512 KB cap before parsing. A recipe is pure data
* applied through `dashboard.workspace.replace`, so it is validated in full here.
*/
async function fetchRecipe(recipeUrl) {
	return parseRecipeBundle(await fetchTextCapped(recipeUrl, GALLERY_RECIPE_MAX_BYTES, "The recipe bundle"));
}
/**
* Install a fetched bundle via the transport. Writes a `pending` registry entry
* (never approved); the operator still approves through the approval gate before
* the widget mounts in its sandbox. Passes only the already-fetched bytes — no URL.
*/
async function installGalleryWidget(transport, bundle) {
	if (!transport) throw new Error("Not connected.");
	await transport.request("dashboard.widget.install", {
		name: bundle.name,
		manifest: bundle.manifest,
		files: bundle.files
	});
}
//#endregion
//#region ../../node_modules/.pnpm/lit-html@3.3.3/node_modules/lit-html/directive-helpers.js
/**
* @license
* Copyright 2020 Google LLC
* SPDX-License-Identifier: BSD-3-Clause
*/ const { I: t$2 } = j$1, r$1 = (o) => void 0 === o.strings;
//#endregion
//#region ../../node_modules/.pnpm/lit-html@3.3.3/node_modules/lit-html/directive.js
/**
* @license
* Copyright 2017 Google LLC
* SPDX-License-Identifier: BSD-3-Clause
*/
const t$1 = {
	ATTRIBUTE: 1,
	CHILD: 2,
	PROPERTY: 3,
	BOOLEAN_ATTRIBUTE: 4,
	EVENT: 5,
	ELEMENT: 6
}, e$2 = (t) => (...e) => ({
	_$litDirective$: t,
	values: e
});
var i = class {
	constructor(t) {}
	get _$AU() {
		return this._$AM._$AU;
	}
	_$AT(t, e, i) {
		this._$Ct = t, this._$AM = e, this._$Ci = i;
	}
	_$AS(t, e) {
		return this.update(t, e);
	}
	update(t, e) {
		return this.render(...e);
	}
};
//#endregion
//#region ../../node_modules/.pnpm/lit-html@3.3.3/node_modules/lit-html/async-directive.js
/**
* @license
* Copyright 2017 Google LLC
* SPDX-License-Identifier: BSD-3-Clause
*/ const s = (i, t) => {
	const e = i._$AN;
	if (void 0 === e) return !1;
	for (const i of e) i._$AO?.(t, !1), s(i, t);
	return !0;
}, o$2 = (i) => {
	let t, e;
	do {
		if (void 0 === (t = i._$AM)) break;
		e = t._$AN, e.delete(i), i = t;
	} while (0 === e?.size);
}, r = (i) => {
	for (let t; t = i._$AM; i = t) {
		let e = t._$AN;
		if (void 0 === e) t._$AN = e = /* @__PURE__ */ new Set();
		else if (e.has(i)) break;
		e.add(i), c(t);
	}
};
function h$1(i) {
	void 0 !== this._$AN ? (o$2(this), this._$AM = i, r(this)) : this._$AM = i;
}
function n$1(i, t = !1, e = 0) {
	const r = this._$AH, h = this._$AN;
	if (void 0 !== h && 0 !== h.size) if (t) if (Array.isArray(r)) for (let i = e; i < r.length; i++) s(r[i], !1), o$2(r[i]);
	else null != r && (s(r, !1), o$2(r));
	else s(this, i);
}
const c = (i) => {
	i.type == t$1.CHILD && (i._$AP ??= n$1, i._$AQ ??= h$1);
};
var f = class extends i {
	constructor() {
		super(...arguments), this._$AN = void 0;
	}
	_$AT(i, t, e) {
		super._$AT(i, t, e), r(this), this.isConnected = i._$AU;
	}
	_$AO(i, t = !0) {
		i !== this.isConnected && (this.isConnected = i, i ? this.reconnected?.() : this.disconnected?.()), t && (s(this, i), o$2(this));
	}
	setValue(t) {
		if (r$1(this._$Ct)) this._$Ct._$AI(t, this);
		else {
			const i = [...this._$Ct._$AH];
			i[this._$Ci] = t, this._$Ct._$AI(i, this, 0);
		}
	}
	disconnected() {}
	reconnected() {}
};
//#endregion
//#region src/boardstate-custom-widget.ts
function bindingByManifestId(widget, bindingId) {
	return widget.bindings?.[bindingId] ?? null;
}
/**
* Wire the parent bridge for one iframe: manifest gating, binding resolution over
* the injected transport, theme tokens, and prompt dispatch. Returns the teardown.
*/
function attachWidgetBridge(params) {
	const { iframe, widget, manifest, context } = params;
	const tabSlug = context.tabSlug ?? "";
	const subscriberId = nextSubscriberId();
	const dispose = mountCustomWidget(iframe, {
		manifest,
		bus: {
			publish: (channel, payload) => publish({
				tabSlug,
				channel,
				fromSubscriberId: subscriberId,
				payload
			}),
			subscribe: (channel, deliver) => subscribe({
				tabSlug,
				channel,
				subscriberId,
				deliver
			})
		},
		getWidgetState: async () => {
			if (!context.transport) throw new Error("Not connected.");
			return createWidgetStateAccessor(context.transport, widget.id).get();
		},
		setWidgetState: async (blob) => {
			if (!context.transport) throw new Error("Not connected.");
			return createWidgetStateAccessor(context.transport, widget.id).set(blob);
		},
		assertBindingAllowed: (bindingId) => {
			const binding = bindingByManifestId(widget, bindingId);
			if (binding?.source === "rpc" && !isRpcMethodAllowed(binding.method ?? "")) return "binding_denied";
			if (binding?.source === "stream" && !isStreamEventAllowed(binding.event ?? "")) return "binding_denied";
			return null;
		},
		resolveBinding: async (bindingId) => {
			const binding = bindingByManifestId(widget, bindingId);
			if (!binding) throw new Error(`binding not configured: ${bindingId}`);
			const result = await resolveBinding(context.transport, binding);
			if ("error" in result) throw new Error(result.error);
			return result.value;
		},
		resolveTheme: context.readThemeTokens ?? readThemeTokensFromRoot,
		confirmPrompt: async (text) => {
			if (context.confirmPrompt) return await context.confirmPrompt(text);
			return typeof window !== "undefined" ? window.confirm(text) : false;
		},
		sendPrompt: async (text) => {
			if (!context.transport) throw new Error("Not connected.");
			await context.transport.request("chat.send", {
				sessionKey: context.sessionKey,
				message: text,
				deliver: false
			});
		}
	});
	return () => {
		dispose();
		unsubscribeAll(tabSlug, subscriberId);
	};
}
/**
* Lit directive owning the iframe's lifecycle: it constructs the sandboxed iframe
* once, attaches the bridge, and tears both down on disconnect. A directive (rather
* than re-rendering an `<iframe>` template) keeps the frame from being recreated on
* every parent render, which would drop bridge state and reload the widget.
*/
var CustomWidgetFrameDirective = class extends f {
	constructor(..._args) {
		super(..._args);
		this.iframe = null;
		this.detach = null;
		this.key = "";
	}
	render(params) {
		const name = params.widget.kind.slice(7);
		const src = widgetAssetUrl(params.context.basePath, name, "index.html");
		const nextKey = `${params.widget.id}::${src}`;
		if (this.iframe && this.key === nextKey) return this.iframe;
		this.detach?.();
		const iframe = document.createElement("iframe");
		iframe.setAttribute("sandbox", "allow-scripts");
		iframe.setAttribute("referrerpolicy", "no-referrer");
		iframe.setAttribute("loading", "lazy");
		iframe.className = "dashboard-widget__frame";
		iframe.title = params.widget.title;
		iframe.src = src;
		iframe.setAttribute("data-test-id", "boardstate-custom-widget-frame");
		this.detach = attachWidgetBridge({
			iframe,
			widget: params.widget,
			manifest: params.manifest,
			context: params.context
		});
		this.iframe = iframe;
		this.key = nextKey;
		return iframe;
	}
	disconnected() {
		this.detach?.();
		this.detach = null;
		this.iframe = null;
		this.key = "";
	}
};
const customWidgetFrame = e$2(CustomWidgetFrameDirective);
/** Renders the sandboxed iframe host for an approved custom widget. */
function renderCustomWidgetHost(params) {
	return b`<div class="dashboard-widget__custom" data-test-id="boardstate-custom-widget">
    ${customWidgetFrame(params)}
  </div>`;
}
//#endregion
//#region src/icons.ts
function glyph(paths) {
	return b`<svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    ${paths}
  </svg>`;
}
const icons = {
	spark: glyph(w`<path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2 2M16 16l2 2M18 6l-2 2M8 16l-2 2" />`),
	x: glyph(w`<path d="M18 6L6 18M6 6l12 12" />`),
	plus: glyph(w`<path d="M12 5v14M5 12h14" />`),
	eyeOff: glyph(w`<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20C5 20 1 12 1 12a18.5 18.5 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19M1 1l22 22" />`),
	chevronRight: glyph(w`<path d="M9 18l6-6-6-6" />`),
	chevronDown: glyph(w`<path d="M6 9l6 6 6-6" />`),
	arrowUpDown: glyph(w`<path d="M7 15l5 5 5-5M7 9l5-5 5 5" />`),
	moreHorizontal: glyph(w`<circle cx="12" cy="12" r="1" /><circle cx="19" cy="12" r="1" /><circle cx="5" cy="12" r="1" />`),
	externalLink: glyph(w`<path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><path d="M15 3h6v6" /><path d="M10 14L21 3" />`),
	clock: glyph(w`<circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />`),
	puzzle: glyph(w`<path d="M4 7h3a1.5 1.5 0 1 0 3 0h3v3a1.5 1.5 0 1 1 0 3v3h-3a1.5 1.5 0 1 0-3 0H4v-3a1.5 1.5 0 1 1 0-3z" />`),
	maximize: glyph(w`<path d="M8 3H5a2 2 0 0 0-2 2v3M16 3h3a2 2 0 0 1 2 2v3M8 21H5a2 2 0 0 1-2-2v-3M16 21h3a2 2 0 0 0 2-2v-3" />`),
	minimize: glyph(w`<path d="M8 3v3a2 2 0 0 1-2 2H3M21 8h-3a2 2 0 0 1-2-2V3M3 16h3a2 2 0 0 1 2 2v3M16 21v-3a2 2 0 0 1 2-2h3" />`)
};
//#endregion
//#region ../../node_modules/.pnpm/lit-html@3.3.3/node_modules/lit-html/directives/ref.js
/**
* @license
* Copyright 2020 Google LLC
* SPDX-License-Identifier: BSD-3-Clause
*/ const e$1 = () => new h();
var h = class {};
const o$1 = /* @__PURE__ */ new WeakMap(), n = e$2(class extends f {
	render(i) {
		return A;
	}
	update(i, [s]) {
		const e = s !== this.G;
		return e && this.rt(void 0), (e || this.lt !== this.ct) && (this.G = s, this.ht = i.options?.host, this.rt(this.ct = i.element)), A;
	}
	rt(t) {
		if (void 0 !== this.G) if (this.isConnected || (t = void 0), "function" == typeof this.G) {
			const i = this.ht ?? globalThis;
			let s = o$1.get(i);
			void 0 === s && (s = /* @__PURE__ */ new WeakMap(), o$1.set(i, s)), void 0 !== s.get(this.G) && this.G.call(this.ht, void 0), s.set(this.G, t), void 0 !== t && this.G.call(this.ht, t);
		} else this.G.value = t;
	}
	get lt() {
		return "function" == typeof this.G ? o$1.get(this.ht ?? globalThis)?.get(this.G) : this.G?.value;
	}
	disconnected() {
		this.lt === this.ct && this.rt(void 0);
	}
	reconnected() {
		this.rt(this.ct);
	}
});
//#endregion
//#region src/strings.ts
/** Full English string table. Its keys define the `BoardstateStrings` surface. */
const en = {
	"common.save": "Save",
	"common.cancel": "Cancel",
	"common.reload": "Reload",
	"common.loading": "Loading…",
	"common.dismiss": "Dismiss",
	"dashboard.header.subtitle": "Your pinned widgets and workspaces.",
	"dashboard.tabs.label": "Workspaces",
	"dashboard.tabs.hidden": "Hidden ({count})",
	"dashboard.error.title": "Couldn’t load your workspace",
	"dashboard.error.subtitle": "Something went wrong reading the workspace document.",
	"dashboard.error.detailSummary": "Error detail",
	"dashboard.empty.onboardingTitle": "No workspaces yet",
	"dashboard.empty.onboardingSubtitle": "Ask the agent to add a workspace tab, or use the CLI.",
	"dashboard.empty.onboardingCommand": "boardstate tab add <name>",
	"dashboard.empty.noVisibleTabs": "All workspace tabs are hidden.",
	"dashboard.empty.tabTitle": "This workspace is empty",
	"dashboard.empty.tabSubtitle": "Ask the agent to add a widget here.",
	"dashboard.onboarding.title": "Add your first workspace",
	"dashboard.onboarding.primary": "Ask the agent to create a workspace tab for you.",
	"dashboard.onboarding.secondary": "Or add one from the CLI:",
	"dashboard.widget.editTitleTitle": "Edit widget title",
	"dashboard.widget.editTitleLabel": "Widget title",
	"dashboard.widget.moveToTabTitle": "Move widget to tab",
	"dashboard.widget.moveToTabEmpty": "There are no other tabs to move this widget to.",
	"dashboard.widget.menu.editTitle": "Edit title",
	"dashboard.widget.menu.moveToTab": "Move to tab",
	"dashboard.widget.menu.hide": "Hide",
	"dashboard.widget.menu.remove": "Remove",
	"dashboard.widget.provenanceChip": "AI",
	"dashboard.widget.provenanceTooltip": "Created by {agent}",
	"dashboard.widget.agentChipTooltip": "Built by {agent}",
	"dashboard.widget.expand": "Expand widget",
	"dashboard.widget.collapse": "Collapse widget",
	"dashboard.widget.moveHandle": "Move widget",
	"dashboard.widget.resizeHandle": "Resize widget",
	"dashboard.widget.menuLabel": "Widget menu",
	"dashboard.widget.errorTitle": "This widget hit an error",
	"dashboard.widget.errorHumane": "The rest of your workspace is unaffected.",
	"dashboard.widget.errorDetailSummary": "Error detail",
	"dashboard.widget.customPlaceholder": "Custom widget",
	"dashboard.widget.customLoading": "Loading widget…",
	"dashboard.widget.unknownKind": "Unknown widget: {kind}",
	"dashboard.widget.approval.title": "Approve this widget?",
	"dashboard.widget.approval.byAgent": "Requested by {agent}",
	"dashboard.widget.approval.byUnknown": "Requested by an agent",
	"dashboard.widget.approval.approve": "Approve",
	"dashboard.widget.approval.reject": "Reject",
	"dashboard.widget.approval.unavailable": "This widget is unavailable.",
	"dashboard.widget.stat.empty": "—",
	"dashboard.widget.markdownEmpty": "Nothing to show yet.",
	"dashboard.widget.table.empty": "No rows to show.",
	"dashboard.widget.table.more": "+{count} more",
	"dashboard.widget.sessions.empty": "No sessions yet.",
	"dashboard.widget.usage.cost": "Cost",
	"dashboard.widget.usage.tokens": "Tokens",
	"dashboard.widget.cron.empty": "No scheduled jobs.",
	"dashboard.widget.cron.next": "Next {time}",
	"dashboard.widget.cron.noNext": "Not scheduled",
	"dashboard.widget.instances.empty": "No connected instances.",
	"dashboard.widget.instances.idle": "idle {duration}",
	"dashboard.widget.activity.empty": "No recent activity.",
	"dashboard.widget.embed.missing": "No URL configured for this embed.",
	"dashboard.widget.embed.blockedExternal": "External embeds are blocked by policy.",
	"dashboard.widget.embed.blockedScheme": "This URL scheme cannot be embedded.",
	"dashboard.widget.chart.empty": "No data to chart.",
	"dashboard.widget.chart.label": "Chart",
	"dashboard.widget.notes.placeholder": "Write a note…",
	"dashboard.widget.notes.readonlyHint": "Connect to the gateway to edit and save notes.",
	"dashboard.widget.actionForm.empty": "This action form has no fields yet.",
	"dashboard.widget.actionForm.submit": "Send",
	"dashboard.widget.actionForm.toolPending": "Submitted — waiting for operator confirmation.",
	"dashboard.widget.actionButton.run": "Run",
	"dashboard.widget.actionButton.invoking": "Invoking…",
	"dashboard.widget.actionButton.pending": "Waiting for operator confirmation…",
	"dashboard.widget.actionButton.confirm": "Confirm",
	"dashboard.widget.actionButton.deny": "Deny",
	"dashboard.widget.actionButton.operatorOnly": "Only the local operator can confirm this action.",
	"dashboard.widget.actionButton.confirmed": "Confirmed.",
	"dashboard.widget.actionButton.denied": "Denied by the operator.",
	"dashboard.widget.actionButton.expired": "The confirmation window expired.",
	"dashboard.widget.actionButton.resultLabel": "Result",
	"dashboard.widget.actionButton.errorLabel": "Error",
	"dashboard.widget.actionButton.disconnected": "Connect to the gateway to run this action.",
	"dashboard.widget.actionButton.misconfigured": "This action is missing a connector or tool.",
	"dashboard.widget.preview.missing": "This preview has no URL yet.",
	"dashboard.widget.preview.blockedExternal": "External previews are disabled by your gateway policy.",
	"dashboard.widget.preview.blockedScheme": "This preview URL uses an unsupported scheme.",
	"dashboard.widget.preview.reload": "Reload preview",
	"dashboard.widget.preview.viewport.desktop": "Desktop",
	"dashboard.widget.preview.viewport.tablet": "Tablet",
	"dashboard.widget.preview.viewport.mobile": "Mobile",
	"dashboard.widget.agentStatus.empty": "No agents yet.",
	"dashboard.widget.agentStatus.busy": "Busy",
	"dashboard.widget.agentStatus.idle": "Idle",
	"dashboard.widget.agentStatus.progress": "{percent}% of budget",
	"dashboard.widget.approvals.empty": "No pending approvals.",
	"dashboard.widget.approvals.approve": "Approve",
	"dashboard.widget.approvals.deny": "Deny",
	"dashboard.widget.approvals.confirm": "Confirm",
	"dashboard.widget.approvals.requestedBy": "Requested by {agent}",
	"dashboard.widget.approvals.kind.widget": "Widget",
	"dashboard.widget.approvals.kind.capability": "Data source",
	"dashboard.widget.approvals.kind.action": "Action",
	"dashboard.widget.approvals.autoConfirm": "Auto-run",
	"dashboard.widget.approvals.autoConfirmHint": "Runs without confirmation each time",
	"dashboard.widget.approvals.scopeLabel": "Agents",
	"dashboard.widget.approvals.scopeAll": "All agents",
	"dashboard.widget.approvals.scopedTo": "Scoped to {agents}",
	"dashboard.widget.approvals.ttlLabel": "Expires in (min)",
	"dashboard.widget.approvals.expiresIn": "Expires in {duration}",
	"dashboard.widget.approvals.expiresSoon": "Expiring…",
	"dashboard.widget.approvals.save": "Save",
	"dashboard.widget.approvals.revoke": "Revoke",
	"dashboard.widget.chat.empty": "Ask the agent to build or change this board…",
	"dashboard.widget.chat.placeholder": "Message the agent…",
	"dashboard.widget.chat.send": "Send",
	"dashboard.widget.chat.stop": "Stop",
	"dashboard.widget.chat.disconnected": "Connect to the gateway to chat with the agent.",
	"dashboard.widget.chat.roleUser": "You",
	"dashboard.widget.chat.roleAssistant": "Agent",
	"dashboard.widget.chat.actionsOne": "1 action",
	"dashboard.widget.chat.actionsMany": "{count} actions",
	"dashboard.widget.chat.building": "building…",
	"dashboard.widget.chat.retrying": "retrying…",
	"dashboard.widget.chat.jumpToLatest": "Jump to latest",
	"dashboard.widget.chat.args": "Arguments",
	"dashboard.widget.chat.result": "Result",
	"dashboard.widget.chat.tool.readBoard": "Read the board",
	"dashboard.widget.chat.tool.createdTab": "Created tab {name}",
	"dashboard.widget.chat.tool.addedWidget": "Added widget {id}",
	"dashboard.widget.chat.approveTitle": "The agent scaffolded widget “{name}”",
	"dashboard.widget.chat.approve": "Approve",
	"dashboard.widget.chat.reject": "Reject",
	"common.close": "Close",
	"common.back": "Back",
	"dashboard.tabs.presence": "{count} viewing",
	"dashboard.tabs.private": "Private — only you can see this tab",
	"dashboard.tabs.groupUser": "You",
	"dashboard.tabs.groupSystem": "System",
	"dashboard.tabs.groupAgent": "{agent}",
	"dashboard.tabs.collapseGroup": "Collapse {group} tabs",
	"dashboard.tabs.expandGroup": "Expand {group} tabs",
	"dashboard.header.fullBleedEnter": "Full-bleed",
	"dashboard.header.fullBleedExit": "Exit full-bleed",
	"dashboard.agentFilter.label": "Agents",
	"dashboard.agentFilter.all": "All",
	"dashboard.widget.ephemeralBadge": "Temporary",
	"dashboard.widget.ephemeralTooltip": "Temporary answer — pin it to keep it here.",
	"dashboard.widget.menu.pin": "Pin",
	"dashboard.widget.blame.createdBy": "Created by {actor}",
	"dashboard.widget.blame.createdByVersion": "Created by {actor} · v{version}",
	"dashboard.widget.blame.logbookLink": "View in logbook",
	"dashboard.history.open": "History",
	"dashboard.history.title": "Workspace history",
	"dashboard.history.subtitle": "Review recent changes, compare against now, and undo the last one.",
	"dashboard.history.empty": "No history yet — changes appear here after your first edit.",
	"dashboard.history.emptyDetail": "Select a version to preview it.",
	"dashboard.history.version": "Version {version}",
	"dashboard.history.latest": "Latest change",
	"dashboard.history.previewTitle": "Snapshot",
	"dashboard.history.previewEmpty": "This tab had no widgets at this point.",
	"dashboard.history.diffTitle": "Changes since this version",
	"dashboard.history.diffEmpty": "Nothing changed since this version.",
	"dashboard.history.restore": "Undo last change",
	"dashboard.history.restoreConfirm": "Undo the most recent change?",
	"dashboard.history.restoreOnlyNewest": "Only the most recent change can be undone.",
	"dashboard.history.actorUnknown": "Unknown",
	"dashboard.history.kind.widget-added": "Added",
	"dashboard.history.kind.widget-removed": "Removed",
	"dashboard.history.kind.widget-moved": "Moved",
	"dashboard.history.kind.widget-retitled": "Retitled",
	"dashboard.history.kind.tab-added": "Tab added",
	"dashboard.history.kind.tab-removed": "Tab removed",
	"dashboard.history.kind.tab-retitled": "Tab retitled",
	"dashboard.history.summary.added": "+{count}",
	"dashboard.history.summary.removed": "−{count}",
	"dashboard.history.summary.moved": "{count} moved",
	"dashboard.history.summary.retitled": "{count} renamed",
	"dashboard.history.summary.tabs": "{count} tabs",
	"dashboard.history.summary.minor": "Other edit",
	"dashboard.history.previewCaption": "Layout at version {version}",
	"dashboard.gallery.open": "Widget gallery",
	"dashboard.gallery.title": "Widget gallery",
	"dashboard.gallery.subtitle": "Browse a widget registry and install a widget from its URL.",
	"dashboard.gallery.urlLabel": "Registry index URL",
	"dashboard.gallery.urlPlaceholder": "https://example.com/widgets/index.json",
	"dashboard.gallery.browse": "Browse",
	"dashboard.gallery.view": "View",
	"dashboard.gallery.install": "Install",
	"dashboard.gallery.empty": "No widgets found at this registry.",
	"dashboard.gallery.capabilities": "Requested capabilities",
	"dashboard.gallery.noCapabilities": "No special capabilities requested.",
	"dashboard.gallery.pendingNote": "Installed widgets stay pending until you approve them, then run sandboxed.",
	"dashboard.gallery.tabWidgets": "Widgets",
	"dashboard.gallery.tabTemplates": "Templates",
	"dashboard.gallery.recipesEmpty": "No templates found at this registry.",
	"dashboard.gallery.recipeNeedsNothing": "Works out of the box — no grants required.",
	"dashboard.gallery.recipeNeedsConnectors": "Needs: {connectors}",
	"dashboard.gallery.recipeNeedsLabel": "This board will ask for these tools",
	"dashboard.gallery.recipeNoGrants": "No external tools — installs ready to use.",
	"dashboard.gallery.recipeReadOnly": "read-only",
	"dashboard.gallery.recipeInstall": "Install template",
	"dashboard.gallery.recipeInstallNote": "Installing imports the board with its grants requested — approve them in the approvals widget to light it up.",
	"dashboard.distribution.export": "Export",
	"dashboard.distribution.exportTitle": "Download this workspace as a JSON file",
	"dashboard.distribution.import": "Import",
	"dashboard.distribution.importTitle": "Import a workspace from a JSON file"
};
/** Interpolate `{name}` placeholders in `template` from `params`. */
function interpolate(template, params) {
	if (!params) return template;
	return template.replace(/\{(\w+)\}/g, (match, key) => Object.hasOwn(params, key) ? params[key] : match);
}
let activeStrings = { ...en };
/** Install a strings override (merged over the English defaults). */
function setBoardstateStrings(strings) {
	activeStrings = strings ? {
		...en,
		...strings
	} : { ...en };
}
/** Resolve a string key against the active table, interpolating `{param}` values. */
function t(key, params) {
	return interpolate(activeStrings[key] ?? en[key] ?? key, params);
}
//#endregion
//#region src/renderers/action-button.ts
/** Stringify a tool result for INERT display (never markup). Objects pretty-print as JSON. */
function formatResult(value) {
	if (value === void 0) return "";
	if (typeof value === "string") return value;
	try {
		return JSON.stringify(value, null, 2);
	} catch {
		return String(value);
	}
}
/**
* The per-widget interactive island. One instance per widget id (keyed in the module
* map below); it holds the invocation phase + the live action subscription and
* re-renders its own subtree with lit's `render()`.
*/
var ActionButtonController = class {
	constructor(widgetId) {
		this.widgetId = widgetId;
		this.root = null;
		this.ctx = null;
		this.widget = null;
		this.phase = { kind: "idle" };
		this.unsubscribe = null;
		this.rootRef = (element) => {
			if (element instanceof HTMLElement) this.mount(element);
			else this.destroy();
		};
		this.onInvoke = () => {
			const actions = this.ctx?.actions;
			if (!actions || !this.widget) return;
			const model = mapActionButton(this.widget);
			if (!model.connector || !model.tool) {
				this.setPhase({
					kind: "error",
					message: t("dashboard.widget.actionButton.misconfigured")
				});
				return;
			}
			this.setPhase({ kind: "running" });
			actions.invoke({
				connector: model.connector,
				tool: model.tool,
				...model.args ? { args: model.args } : {}
			}).then((outcome) => {
				this.setPhase(outcome.kind === "pending" ? {
					kind: "pending",
					id: outcome.id,
					expiresAt: outcome.expiresAt
				} : {
					kind: "result",
					value: outcome.result
				});
			}).catch((err) => {
				this.setPhase({
					kind: "error",
					message: err instanceof Error ? err.message : String(err)
				});
			});
		};
		this.onConfirm = (id) => {
			const confirm = this.ctx?.actions?.confirm;
			if (!confirm) return;
			this.setPhase({ kind: "running" });
			confirm(id).then(({ result }) => this.setPhase({
				kind: "result",
				value: result
			})).catch((err) => {
				this.setPhase({
					kind: "error",
					message: err instanceof Error ? err.message : String(err)
				});
			});
		};
		this.onDeny = (id) => {
			const deny = this.ctx?.actions?.deny;
			if (!deny) return;
			deny(id).then(() => this.setPhase({ kind: "denied" })).catch((err) => {
				this.setPhase({
					kind: "error",
					message: err instanceof Error ? err.message : String(err)
				});
			});
		};
	}
	/** Absorb the latest render context/widget (parent re-render) and refresh the island. */
	setContext(ctx, widget) {
		this.ctx = ctx;
		this.widget = widget;
		if (this.root) this.renderIsland();
	}
	mount(element) {
		this.root = element;
		this.unsubscribe?.();
		this.unsubscribe = null;
		this.phase = { kind: "idle" };
		this.renderIsland();
		const actions = this.ctx?.actions;
		if (actions) this.unsubscribe = actions.subscribe((change) => this.onActionChange(change));
	}
	destroy() {
		this.unsubscribe?.();
		this.unsubscribe = null;
		this.root = null;
		controllers$1.delete(this.widgetId);
	}
	/** React to a pending-action lifecycle change that concerns THIS button's parked action. */
	onActionChange(change) {
		if (this.phase.kind !== "pending" || change.id !== this.phase.id) return;
		if (change.status === "confirmed") this.phase = { kind: "confirmed" };
		else if (change.status === "denied") this.phase = { kind: "denied" };
		else if (change.status === "expired") this.phase = { kind: "expired" };
		else return;
		this.renderIsland();
	}
	setPhase(phase) {
		this.phase = phase;
		this.renderIsland();
	}
	renderIsland() {
		if (!this.root) return;
		D(this.template(), this.root);
	}
	template() {
		const actions = this.ctx?.actions;
		const label = (this.widget ? mapActionButton(this.widget) : null)?.label ?? t("dashboard.widget.actionButton.run");
		const busy = this.phase.kind === "running" || this.phase.kind === "pending";
		return b`
      <div class="dashboard-action-button" data-test-id="dashboard-action-button">
        <button
          class="bs-btn bs-btn--small bs-btn--primary dashboard-action-button__invoke"
          type="button"
          data-test-id="dashboard-action-button-invoke"
          ?disabled=${!actions || busy}
          @click=${this.onInvoke}
        >
          ${label}
        </button>
        ${!actions ? b`<div
                class="dashboard-action-button__hint"
                data-test-id="dashboard-action-button-disconnected"
              >
                ${t("dashboard.widget.actionButton.disconnected")}
              </div>` : this.renderStatus()}
      </div>
    `;
	}
	renderStatus() {
		switch (this.phase.kind) {
			case "idle": return A;
			case "running": return b`<div class="dashboard-action-button__status" data-status="running">
          ${t("dashboard.widget.actionButton.invoking")}
        </div>`;
			case "pending": return this.renderPending(this.phase.id);
			case "confirmed": return b`<div
          class="dashboard-action-button__status"
          data-status="confirmed"
          data-test-id="dashboard-action-button-confirmed"
        >
          ${t("dashboard.widget.actionButton.confirmed")}
        </div>`;
			case "denied": return b`<div
          class="dashboard-action-button__status"
          data-status="denied"
          data-test-id="dashboard-action-button-denied"
        >
          ${t("dashboard.widget.actionButton.denied")}
        </div>`;
			case "expired": return b`<div
          class="dashboard-action-button__status"
          data-status="expired"
          data-test-id="dashboard-action-button-expired"
        >
          ${t("dashboard.widget.actionButton.expired")}
        </div>`;
			case "result": return b`<div class="dashboard-action-button__result" data-status="result">
          <div class="dashboard-action-button__result-label">
            ${t("dashboard.widget.actionButton.resultLabel")}
          </div>
          <pre
            class="dashboard-action-button__result-body"
            data-test-id="dashboard-action-button-result"
          >
${formatResult(this.phase.value)}</pre>
        </div>`;
			case "error": return b`<div
          class="dashboard-action-button__error"
          role="alert"
          data-test-id="dashboard-action-button-error"
        >
          <span class="dashboard-action-button__result-label"
            >${t("dashboard.widget.actionButton.errorLabel")}</span
          >
          <span class="dashboard-action-button__error-message">${this.phase.message}</span>
        </div>`;
		}
	}
	/** The parked-mutation row: "waiting for operator" + confirm/deny (operator only). */
	renderPending(id) {
		const canConfirm = Boolean(this.ctx?.actions?.confirm && this.ctx?.actions?.deny);
		return b`
      <div
        class="dashboard-action-button__pending"
        data-status="pending"
        data-test-id="dashboard-action-button-pending"
      >
        <span class="dashboard-action-button__status-text"
          >${t("dashboard.widget.actionButton.pending")}</span
        >
        ${canConfirm ? b`<span class="dashboard-action-button__pending-actions">
                <button
                  class="bs-btn bs-btn--small bs-btn--primary"
                  type="button"
                  data-test-id="dashboard-action-button-confirm"
                  @click=${() => this.onConfirm(id)}
                >
                  ${t("dashboard.widget.actionButton.confirm")}
                </button>
                <button
                  class="bs-btn bs-btn--small"
                  type="button"
                  data-test-id="dashboard-action-button-deny"
                  @click=${() => this.onDeny(id)}
                >
                  ${t("dashboard.widget.actionButton.deny")}
                </button>
              </span>` : b`<span
                class="dashboard-action-button__operator-only"
                data-test-id="dashboard-action-button-operator-only"
                >${t("dashboard.widget.actionButton.operatorOnly")}</span
              >`}
      </div>
    `;
	}
};
/** One live controller per widget id. Created lazily; removed on the widget's unmount. */
const controllers$1 = /* @__PURE__ */ new Map();
/**
* Renders builtin:action-button. The renderer stays a pure function returning the
* island's container; the `ActionButtonController` (keyed by widget id) owns the
* invocation lifecycle and its own render loop, hydrated via the `ref` callback.
*/
function renderActionButton(widget, _value, ctx) {
	let controller = controllers$1.get(widget.id);
	if (!controller) {
		controller = new ActionButtonController(widget.id);
		controllers$1.set(widget.id, controller);
	}
	controller.setContext(ctx, widget);
	return b`<div class="dashboard-action-button-host" ${n(controller.rootRef)}></div>`;
}
//#endregion
//#region src/renderers/action-form.ts
function renderField(field) {
	const control = field.type === "select" ? b`<select class="dashboard-action-form__control" name=${field.name}>
          ${(field.options ?? []).map((option) => b`<option value=${option}>${option}</option>`)}
        </select>` : b`<input
          class="dashboard-action-form__control"
          type=${field.type === "number" ? "number" : "text"}
          name=${field.name}
          maxlength=${field.maxLength ?? 200}
        />`;
	return b`<label class="dashboard-action-form__field">
    <span class="dashboard-action-form__label">${field.label}</span>
    ${control}
  </label>`;
}
/**
* Submit a `tool`-mode form: the coerced field values become the tool ARGS (via
* `argsFrom`) and go through the SAME `dashboard.action.invoke` seam the action-button
* uses — no template interpolation, no new dispatch privilege. A readOnly tool executes
* (the form resets); a mutation PARKS as an operator-confirmed pending action, surfaced
* on the shared toast; a rejection (ungranted/revoked/rate-limited) surfaces there too.
*/
function submitTool(model, widget, values, ctx, form) {
	if (!ctx.actions || !model.connector || !model.tool) return;
	const args = buildActionToolArgs(model, values);
	ctx.actions.invoke({
		connector: model.connector,
		tool: model.tool,
		args
	}).then((outcome) => {
		if (outcome.kind === "pending") ctx.onActionError?.(t("dashboard.widget.actionForm.toolPending"));
		form.reset();
	}).catch((err) => {
		ctx.onActionError?.(err instanceof Error ? err.message : String(err));
	});
}
/** Renders the action-form builtin. Submit interpolates + dispatches through the shared gate. */
function renderActionForm(widget, _value, ctx) {
	const model = mapActionForm(widget);
	if (model.fields.length === 0 || !model.template) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.actionForm.empty")}
    </div>`;
	const readValues = (form) => {
		const values = {};
		for (const field of model.fields) {
			const control = form.elements.namedItem(field.name);
			values[field.name] = control && "value" in control ? String(control.value ?? "") : "";
		}
		return values;
	};
	const onSubmit = (event) => {
		event.preventDefault();
		const form = event.currentTarget;
		const values = readValues(form);
		if (model.mode === "tool") {
			submitTool(model, widget, values, ctx, form);
			return;
		}
		const text = buildActionFormPrompt(model, values);
		if (!text.trim() || !ctx.dispatchPrompt) return;
		ctx.dispatchPrompt({
			widgetKey: `builtin:action-form:${widget.id}`,
			text
		}).then((outcome) => {
			if (outcome === "sent") form.reset();
		}).catch((err) => {
			ctx.onActionError?.(err instanceof Error ? err.message : String(err));
		});
	};
	return b`
    <form class="dashboard-action-form" data-test-id="dashboard-action-form" @submit=${onSubmit}>
      ${model.fields.map(renderField)}
      <button
        class="bs-btn bs-btn--small bs-btn--primary dashboard-action-form__submit"
        type="submit"
      >
        ${model.buttonLabel ?? t("dashboard.widget.actionForm.submit")}
      </button>
    </form>
    ${(model.mode === "tool" ? ctx.actions : ctx.dispatchPrompt) ? A : b`<span hidden data-test-id="dashboard-action-form-inert"></span>`}
  `;
}
//#endregion
//#region src/renderers/format.ts
/** Format a cost as USD, e.g. `3.2` → `$3.20`. */
function formatCost(cost) {
	const value = Number.isFinite(cost) ? cost : 0;
	return new Intl.NumberFormat("en-US", {
		style: "currency",
		currency: "USD"
	}).format(value);
}
/** Format a token count compactly, e.g. `1234567` → `1.2M`. */
function formatTokens(tokens) {
	const value = Number.isFinite(tokens) ? tokens : 0;
	return new Intl.NumberFormat("en-US", {
		notation: "compact",
		maximumFractionDigits: 1
	}).format(value);
}
/** Format an epoch-ms timestamp as a short local date/time. */
function formatDateTimeMs(ms) {
	if (!Number.isFinite(ms)) return "";
	try {
		return new Intl.DateTimeFormat("en-US", {
			month: "short",
			day: "numeric",
			hour: "numeric",
			minute: "2-digit"
		}).format(new Date(ms));
	} catch {
		return new Date(ms).toISOString();
	}
}
/** Format a duration in ms as a compact human string, e.g. `90000` → `1m 30s`. */
function formatMs(ms) {
	if (!Number.isFinite(ms) || ms < 0) return "";
	const totalSeconds = Math.round(ms / 1e3);
	if (totalSeconds < 60) return `${totalSeconds}s`;
	const minutes = Math.floor(totalSeconds / 60);
	const seconds = totalSeconds % 60;
	if (minutes < 60) return seconds ? `${minutes}m ${seconds}s` : `${minutes}m`;
	const hours = Math.floor(minutes / 60);
	const remMinutes = minutes % 60;
	return remMinutes ? `${hours}h ${remMinutes}m` : `${hours}h`;
}
//#endregion
//#region src/renderers/activity.ts
function statusClass$1(status) {
	if (status === "ok") return "dashboard-badge--ok";
	if (status === "error") return "dashboard-badge--error";
	return "dashboard-badge--muted";
}
function renderActivity(widget, value) {
	const model = mapActivity(widget, value);
	if (model.entries.length === 0) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.activity.empty")}
    </div>`;
	return b`
    <ul class="dashboard-feed" data-test-id="dashboard-activity">
      ${model.entries.map((entry) => b`
          <li class="dashboard-feed__row">
            <div class="dashboard-feed__head">
              <span class="dashboard-feed__title">${entry.title}</span>
              ${entry.status ? b`<span class="dashboard-badge ${statusClass$1(entry.status)}"
                      >${entry.status}</span
                    >` : A}
              ${entry.ts !== null ? b`<span class="dashboard-feed__time">${formatDateTimeMs(entry.ts)}</span>` : A}
            </div>
            ${entry.detail ? b`<div class="dashboard-feed__detail">${entry.detail}</div>` : A}
          </li>
        `)}
    </ul>
  `;
}
//#endregion
//#region src/renderers/agent-status.ts
function renderAgentStatus(widget, value) {
	const model = mapAgentStatus(widget, value);
	if (model.rows.length === 0) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.agentStatus.empty")}
    </div>`;
	return b`
    <ul class="dashboard-list dashboard-agent-status" data-test-id="dashboard-agent-status">
      ${model.rows.map((row) => b`
          <li class="dashboard-list__row">
            <span
              class="dashboard-dot ${row.active ? "dashboard-dot--live" : ""}"
              aria-hidden="true"
            ></span>
            <span class="dashboard-list__label">${row.label}</span>
            <span
              class="dashboard-badge ${row.active ? "dashboard-badge--ok" : "dashboard-badge--muted"}"
            >
              ${row.active ? t("dashboard.widget.agentStatus.busy") : t("dashboard.widget.agentStatus.idle")}
            </span>
            ${row.task ? b`<span class="dashboard-list__meta">${row.task}</span>` : A}
            ${row.progress !== null ? b`<span class="dashboard-list__meta"
                    >${t("dashboard.widget.agentStatus.progress", { percent: String(Math.round(row.progress * 100)) })}</span
                  >` : A}
          </li>
        `)}
    </ul>
  `;
}
//#endregion
//#region src/renderers/approvals.ts
/** The badge label for an approval row's kind (widget / data source / action). */
function kindLabel(kind) {
	if (kind === "capability") return t("dashboard.widget.approvals.kind.capability");
	if (kind === "action") return t("dashboard.widget.approvals.kind.action");
	return t("dashboard.widget.approvals.kind.widget");
}
/** Collect the values of the ticked checkboxes matching `selector` inside THIS row. */
function checkedValues(event, selector) {
	const row = event.currentTarget?.closest("li");
	if (!row) return [];
	return [...row.querySelectorAll(selector)].filter((box) => box.checked).map((box) => box.value);
}
/** Read the row's optional TTL input (minutes) and turn it into a future ISO instant. */
function readTtl(event) {
	const input = (event.currentTarget?.closest("li"))?.querySelector("input.dashboard-approvals__ttl");
	const minutes = input && input.value.trim() !== "" ? Number(input.value) : NaN;
	if (!Number.isFinite(minutes) || minutes <= 0) return;
	return new Date(Date.now() + minutes * 6e4).toISOString();
}
/**
* Build the operator's decision options from a capability row's controls (#62/#64). For a
* row WITH tool ticks, `tools` is always present (even empty — unticking all grants
* nothing, never approve-all); a data-only row carries just the optional TTL.
*/
function collectCapabilityOptions(event, hasTools) {
	const expiresAt = readTtl(event);
	if (!hasTools) return expiresAt !== void 0 ? { expiresAt } : {};
	const tools = checkedValues(event, "input.dashboard-approvals__grant");
	const autoConfirm = checkedValues(event, "input.dashboard-approvals__auto");
	return {
		tools,
		...autoConfirm.length ? { autoConfirm: autoConfirm.filter((id) => tools.includes(id)) } : {},
		...expiresAt !== void 0 ? { expiresAt } : {}
	};
}
/** A coarse "expires in 2h 5m" label from an ISO instant (refreshed on each re-render). */
function expiresLabel(expiresAt) {
	const remaining = Date.parse(expiresAt) - Date.now();
	if (Number.isNaN(remaining) || remaining <= 0) return t("dashboard.widget.approvals.expiresSoon");
	const minutes = Math.round(remaining / 6e4);
	const hours = Math.floor(minutes / 60);
	return t("dashboard.widget.approvals.expiresIn", { duration: hours > 0 ? `${hours}h ${minutes % 60}m` : `${minutes}m` });
}
/**
* The per-agent scope line for a capability row (SPEC §17.3, #59). A scoped grant shows
* exactly which agents may use its tools; an unscoped grant reads "All agents" so the
* operator sees the (permissive) default plainly rather than an ambiguous blank.
*/
function renderScope(item) {
	const agents = item.agents ?? [];
	const summary = agents.length > 0 ? t("dashboard.widget.approvals.scopedTo", { agents: agents.join(", ") }) : t("dashboard.widget.approvals.scopeAll");
	return b`<span
    class="dashboard-approvals__scope"
    data-test-id="dashboard-approvals-scope"
    data-agents=${agents.join(",")}
    >${t("dashboard.widget.approvals.scopeLabel")}: ${summary}</span
  >`;
}
/** The per-tool grant + auto-confirm control list for a capability row. */
function renderToolControls(item) {
	const tools = item.tools ?? [];
	const auto = new Set(item.autoConfirm ?? []);
	return b`<ul class="dashboard-approvals__tools" data-test-id="dashboard-approvals-tools">
    ${tools.map((tool) => b`<li>
          <label class="dashboard-approvals__grant-label"
            ><input type="checkbox" class="dashboard-approvals__grant" value=${tool} checked /><span
              >${tool}</span
            ></label
          >
          <label
            class="dashboard-approvals__auto-label"
            title=${t("dashboard.widget.approvals.autoConfirmHint")}
            ><input
              type="checkbox"
              class="dashboard-approvals__auto"
              value=${tool}
              ?checked=${auto.has(tool)}
            /><span>${t("dashboard.widget.approvals.autoConfirm")}</span></label
          >
        </li>`)}
  </ul>`;
}
function renderApprovals(widget, _value, ctx) {
	const source = ctx.approvals;
	const model = mapApprovals(widget, source);
	if (model.items.length === 0) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.approvals.empty")}
    </div>`;
	return b`
    <ul class="dashboard-list dashboard-approvals" data-test-id="dashboard-approvals">
      ${model.items.map((item) => {
		const isCapability = item.kind === "capability";
		const hasTools = isCapability && (item.tools ?? []).length > 0;
		const affirmLabel = item.granted ? t("dashboard.widget.approvals.save") : item.kind === "action" ? t("dashboard.widget.approvals.confirm") : t("dashboard.widget.approvals.approve");
		const affirm = (event) => {
			if (!isCapability) {
				source?.onDecide(item, "approve");
				return;
			}
			const options = collectCapabilityOptions(event, hasTools);
			if (Object.keys(options).length > 0) source?.onDecide(item, "approve", options);
			else source?.onDecide(item, "approve");
		};
		const denyLabel = item.granted ? t("dashboard.widget.approvals.revoke") : t("dashboard.widget.approvals.deny");
		return b`
          <li
            class="dashboard-list__row ${item.granted ? "dashboard-approvals__row--granted" : ""}"
          >
            <span class="dashboard-badge dashboard-badge--muted">${kindLabel(item.kind)}</span>
            <span class="dashboard-list__label">${item.title}</span>
            ${item.detail ? b`<span class="dashboard-list__meta">${item.detail}</span>` : item.requestedBy ? b`<span class="dashboard-list__meta"
                      >${t("dashboard.widget.approvals.requestedBy", { agent: item.requestedBy })}</span
                    >` : A}
            ${item.expiresAt ? b`<span
                    class="dashboard-approvals__countdown"
                    data-test-id="dashboard-approvals-countdown"
                    >${expiresLabel(item.expiresAt)}</span
                  >` : A}
            ${hasTools ? renderToolControls(item) : A}
            ${isCapability ? renderScope(item) : A}
            ${isCapability ? b`<label class="dashboard-approvals__ttl-label"
                    >${t("dashboard.widget.approvals.ttlLabel")}
                    <input
                      type="number"
                      min="1"
                      class="dashboard-approvals__ttl"
                      data-test-id="dashboard-approvals-ttl"
                  /></label>` : A}
            <span class="dashboard-approvals__actions">
              <button
                class="bs-btn bs-btn--small bs-btn--primary"
                type="button"
                data-test-id="dashboard-approvals-approve"
                @click=${affirm}
              >
                ${affirmLabel}
              </button>
              <button
                class="bs-btn bs-btn--small"
                type="button"
                data-test-id="dashboard-approvals-deny"
                @click=${() => source?.onDecide(item, "reject")}
              >
                ${denyLabel}
              </button>
            </span>
          </li>
        `;
	})}
    </ul>
  `;
}
//#endregion
//#region src/renderers/chart.ts
const VIEW_W = 100;
const VIEW_H = 40;
const PAD = 2;
/** Map a value onto the [PAD, VIEW_H-PAD] band, flat-lining a zero-range series. */
function yScale(v, min, max) {
	const span = max - min;
	if (span <= 0) return VIEW_H / 2;
	const norm = (v - min) / span;
	return VIEW_H - PAD - norm * (VIEW_H - PAD * 2);
}
/** X position for the i-th of n points across the padded width. */
function xScale(i, n) {
	if (n <= 1) return VIEW_W / 2;
	return PAD + i / (n - 1) * (VIEW_W - PAD * 2);
}
function linePoints(values, min, max) {
	return values.map((v, i) => `${xScale(i, values.length)},${yScale(v, min, max)}`).join(" ");
}
const numberFormat = new Intl.NumberFormat("en-US", {
	notation: "compact",
	maximumFractionDigits: 1
});
function formatValue(v) {
	return Number.isFinite(v) ? numberFormat.format(v) : "";
}
function sparkTrend(values) {
	if (values.length < 2) return "flat";
	const first = values[0];
	const last = values[values.length - 1];
	return last > first ? "up" : last < first ? "down" : "flat";
}
function drawLine(model) {
	return w`<polyline
    class="dashboard-chart__line"
    fill="none"
    points=${linePoints(model.values, model.min, model.max)}
  />`;
}
function drawArea(model) {
	const points = linePoints(model.values, model.min, model.max);
	const first = xScale(0, model.values.length);
	const last = xScale(model.values.length - 1, model.values.length);
	const base = VIEW_H - PAD;
	return w`<g>
    <polygon class="dashboard-chart__area" points=${`${first},${base} ${points} ${last},${base}`} />
    <polyline class="dashboard-chart__line" fill="none" points=${points} />
  </g>`;
}
function drawBars(model) {
	const n = model.values.length;
	const slot = (VIEW_W - PAD * 2) / n;
	const gap = slot > 3 ? Math.min(1, slot * .2) : 0;
	const width = Math.max(slot - gap, .5);
	const base = VIEW_H - PAD;
	return w`<g class="dashboard-chart__bars">
    ${model.values.map((v, i) => {
		const y = yScale(v, model.min, model.max);
		return w`<rect x=${PAD + i * slot + gap / 2} y=${y} width=${width} height=${Math.max(base - y, 0)} />`;
	})}
  </g>`;
}
/** Gauge — a 180° arc with a needle at the value's position in [min,max]. */
function drawGauge(model, props) {
	const current = model.values.length ? model.values[model.values.length - 1] : 0;
	const lo = toFiniteNumber(props.min) ?? Math.min(model.min, 0);
	const span = (toFiniteNumber(props.max) ?? Math.max(model.max, current)) - lo;
	const frac = span > 0 ? Math.min(Math.max((current - lo) / span, 0), 1) : 0;
	const cx = VIEW_W / 2;
	const cy = VIEW_H - PAD;
	const r = Math.min(VIEW_W / 2, VIEW_H) - PAD;
	const polar = (fraction) => {
		const angle = Math.PI - fraction * Math.PI;
		return {
			x: cx + r * Math.cos(angle),
			y: cy - r * Math.sin(angle)
		};
	};
	const start = polar(0);
	const end = polar(1);
	const value = polar(frac);
	return w`<g class="dashboard-chart__gauge">
    <path class="dashboard-chart__gauge-track" fill="none" d=${`M ${start.x} ${start.y} A ${r} ${r} 0 0 1 ${end.x} ${end.y}`} />
    <path class="dashboard-chart__gauge-fill" fill="none" d=${`M ${start.x} ${start.y} A ${r} ${r} 0 0 1 ${value.x} ${value.y}`} />
    <line class="dashboard-chart__gauge-needle" x1=${cx} y1=${cy} x2=${value.x} y2=${value.y} />
  </g>`;
}
function drawSparkline(model) {
	const n = model.values.length;
	const trend = sparkTrend(model.values);
	if (n < 2) return w`<g class="dashboard-chart__spark dashboard-chart__spark--${trend}">
      <circle class="dashboard-chart__spark-dot" cx=${xScale(0, n)} cy=${yScale(model.values[0] ?? 0, model.min, model.max)} r="1.5" />
    </g>`;
	return w`<g class="dashboard-chart__spark dashboard-chart__spark--${trend}">
    <polyline class="dashboard-chart__line" fill="none" points=${linePoints(model.values, model.min, model.max)} />
  </g>`;
}
/** Only the cartesian types carry a y-axis; gauge and sparkline never do. */
function hasAxes(type) {
	return type === "line" || type === "area" || type === "bar";
}
/** Detail-mode gridlines — three faint horizontals at the min/mid/max bands. */
function drawGrid() {
	return w`<g class="dashboard-chart__grid">
    ${[
		PAD,
		VIEW_H / 2,
		VIEW_H - PAD
	].map((y) => w`<line x1=${PAD} y1=${y} x2=${VIEW_W - PAD} y2=${y} />`)}
  </g>`;
}
function drawTips(model) {
	const n = model.values.length;
	if (model.type === "bar") {
		const slot = (VIEW_W - PAD * 2) / n;
		return w`<g class="dashboard-chart__tips">
      ${model.values.map((v, i) => w`<rect class="dashboard-chart__tip" x=${PAD + i * slot} y=${PAD} width=${slot} height=${VIEW_H - PAD * 2}><title>${formatValue(v)}</title></rect>`)}
    </g>`;
	}
	if (model.type === "gauge") {
		const current = n ? model.values[n - 1] : 0;
		return w`<g class="dashboard-chart__tips">
      <rect class="dashboard-chart__tip" x=${PAD} y=${PAD} width=${VIEW_W - PAD * 2} height=${VIEW_H - PAD * 2}><title>${formatValue(current)}</title></rect>
    </g>`;
	}
	return w`<g class="dashboard-chart__tips">
    ${model.values.map((v, i) => w`<circle class="dashboard-chart__tip" cx=${xScale(i, n)} cy=${yScale(v, model.min, model.max)} r="2.5"><title>${formatValue(v)}</title></circle>`)}
  </g>`;
}
function drawBase(model, props) {
	switch (model.type) {
		case "bar": return drawBars(model);
		case "area": return drawArea(model);
		case "gauge": return drawGauge(model, props);
		case "sparkline": return drawSparkline(model);
		default: return drawLine(model);
	}
}
function drawChart(model, props) {
	const base = drawBase(model, props);
	if (!model.detail || model.type === "sparkline") return base;
	return w`<g>
    ${hasAxes(model.type) ? drawGrid() : A}
    ${base}
    ${drawTips(model)}
  </g>`;
}
function renderChart(widget, value) {
	const model = mapChart(widget, value);
	if (model.values.length === 0) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.chart.empty")}
    </div>`;
	const props = widgetProps(widget);
	const detail = model.detail && model.type !== "sparkline";
	const axes = detail && hasAxes(model.type);
	const sparkValue = model.type === "sparkline" && model.label;
	const detailClass = detail ? " dashboard-chart--detail" : "";
	return b`
    <div class="dashboard-chart dashboard-chart--${model.type}${detailClass}">
      <svg
        class="dashboard-chart__svg"
        viewBox="0 0 ${VIEW_W} ${VIEW_H}"
        preserveAspectRatio="none"
        role="img"
        aria-label=${widget.title ?? t("dashboard.widget.chart.label")}
        data-test-id="dashboard-chart"
      >
        ${drawChart(model, props)}
      </svg>
      ${axes ? b`<span class="dashboard-chart__axis dashboard-chart__axis--max"
                >${formatValue(model.max)}</span
              ><span class="dashboard-chart__axis dashboard-chart__axis--min"
                >${formatValue(model.min)}</span
              >` : A}
      ${sparkValue ? b`<span
              class="dashboard-chart__spark-value dashboard-chart__spark-value--${sparkTrend(model.values)}"
              >${formatValue(model.values[model.values.length - 1] ?? 0)}</span
            >` : A}
    </div>
  `;
}
//#endregion
//#region ../../node_modules/.pnpm/lit-html@3.3.3/node_modules/lit-html/directives/unsafe-html.js
/**
* @license
* Copyright 2017 Google LLC
* SPDX-License-Identifier: BSD-3-Clause
*/ var e = class extends i {
	constructor(i) {
		if (super(i), this.it = A, i.type !== t$1.CHILD) throw Error(this.constructor.directiveName + "() can only be used in child bindings");
	}
	render(r) {
		if (r === A || null == r) return this._t = void 0, this.it = r;
		if (r === E) return r;
		if ("string" != typeof r) throw Error(this.constructor.directiveName + "() called with a non-string value");
		if (r === this.it) return this._t;
		this.it = r;
		const s = [r];
		return s.raw = s, this._t = {
			_$litType$: this.constructor.resultType,
			strings: s,
			values: []
		};
	}
};
e.directiveName = "unsafeHTML", e.resultType = 1;
const o = e$2(e);
//#endregion
//#region src/markdown.ts
/** HTML-escape the five significant characters. */
function escapeHtml(text) {
	return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}
/** True for an absolute http(s) URL (the only href scheme links may carry). */
function isSafeHref(url) {
	return /^https?:\/\//i.test(url.trim());
}
/** Apply inline markdown to an already-escaped line: code, links, bold, italic. */
function renderInline(escaped) {
	let out = escaped;
	out = out.replace(/`([^`]+)`/g, (_match, code) => `<code>${code}</code>`);
	out = out.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (match, text, url) => isSafeHref(url) ? `<a href="${url}" rel="noopener noreferrer">${text}</a>` : match);
	out = out.replace(/\*\*([^*]+)\*\*/g, (_match, inner) => `<strong>${inner}</strong>`);
	out = out.replace(/(^|[^*])\*([^*]+)\*/g, (_m, lead, inner) => `${lead}<em>${inner}</em>`);
	out = out.replace(/(^|[^_])_([^_]+)_/g, (_m, lead, inner) => `${lead}<em>${inner}</em>`);
	return out;
}
/** Render one non-list block (heading / blockquote / paragraph). */
function renderBlock(block) {
	const lines = block.split("\n");
	const heading = /^(#{1,6})\s+(.*)$/.exec(lines[0] ?? "");
	if (heading && lines.length === 1) {
		const level = heading[1].length;
		return `<h${level}>${renderInline(escapeHtml(heading[2]))}</h${level}>`;
	}
	if (lines.every((line) => line.startsWith(">"))) return `<blockquote>${lines.map((line) => renderInline(escapeHtml(line.replace(/^>\s?/, "")))).join("<br>")}</blockquote>`;
	return `<p>${lines.map((line) => renderInline(escapeHtml(line))).join("<br>")}</p>`;
}
/** Render a bullet (`-`/`*`) or ordered (`1.`) list block. */
function renderList(block, ordered) {
	const items = block.split("\n").map((line) => line.replace(ordered ? /^\s*\d+\.\s+/ : /^\s*[-*]\s+/, "")).map((item) => `<li>${renderInline(escapeHtml(item))}</li>`).join("");
	return ordered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`;
}
function isUnorderedList(block) {
	return block.split("\n").every((line) => /^\s*[-*]\s+/.test(line));
}
function isOrderedList(block) {
	return block.split("\n").every((line) => /^\s*\d+\.\s+/.test(line));
}
/** Render a fenced code block (```) as escaped `<pre><code>`. */
function renderFence(lines) {
	return `<pre><code>${escapeHtml(lines.join("\n"))}</code></pre>`;
}
/**
* Convert a markdown source string into sanitized, allowlist-only HTML. Safe to
* inject with `unsafeHTML`: every code path escapes text before wrapping it in one
* of the permitted tags.
*/
function toSanitizedMarkdownHtml(source) {
	const rawLines = source.replace(/\r\n?/g, "\n").split("\n");
	const html = [];
	let paragraph = [];
	const flushParagraph = () => {
		if (paragraph.length === 0) return;
		const block = paragraph.join("\n");
		if (isUnorderedList(block)) html.push(renderList(block, false));
		else if (isOrderedList(block)) html.push(renderList(block, true));
		else html.push(renderBlock(block));
		paragraph = [];
	};
	for (let i = 0; i < rawLines.length; i += 1) {
		const line = rawLines[i];
		if (line.startsWith("```")) {
			flushParagraph();
			const fence = [];
			i += 1;
			while (i < rawLines.length && !rawLines[i].startsWith("```")) {
				fence.push(rawLines[i]);
				i += 1;
			}
			html.push(renderFence(fence));
			continue;
		}
		if (line.trim() === "") {
			flushParagraph();
			continue;
		}
		paragraph.push(line);
	}
	flushParagraph();
	return html.join("\n");
}
//#endregion
//#region src/renderers/chat-model.ts
/** The mark a single tool call contributes to its group chip. */
function chatToolMark(call) {
	if (call.status === "ok") return "ok";
	if (call.status === "error") return "error";
	return "pending";
}
function newTurn(turnId) {
	return {
		turn: {
			turnId,
			items: [],
			status: "streaming"
		},
		textById: /* @__PURE__ */ new Map(),
		callById: /* @__PURE__ */ new Map()
	};
}
/** The last render item of a turn, or undefined when empty. */
function lastItem(turn) {
	return turn.items[turn.items.length - 1];
}
/**
* Mark every still-pending retryable error in the turn as superseded. Called when a
* continuation event (more text / another tool call / another error) arrives after
* an error: the turn kept going, so the error WAS retried — the UI shows "retrying…"
* instead of a final failure.
*/
function markErrorsRetried(turn) {
	for (const item of turn.items) if (item.kind === "error" && item.retryable && !item.superseded) item.superseded = true;
}
/** Get (or lazily create) the working turn for an event's turnId. */
function turnFor(work, order, turnId) {
	let entry = work.get(turnId);
	if (!entry) {
		entry = newTurn(turnId);
		work.set(turnId, entry);
		order.push(turnId);
	}
	return entry;
}
/** Append a tool call to the trailing tools group, or open a new one (grouping breaks on text/error). */
function appendToolCall(turn, call) {
	const tail = lastItem(turn);
	if (tail && tail.kind === "tools") {
		tail.calls.push(call);
		return;
	}
	turn.items.push({
		kind: "tools",
		calls: [call]
	});
}
/**
* Fold a raw `AgentStreamEvent[]` into ordered assistant turns. Pure and total: any
* out-of-order, duplicate, or orphaned event is absorbed rather than thrown. Turns
* keep first-seen order; a turnless `error` attaches to the most recent turn (or a
* synthetic empty-id turn when the stream opens with one).
*/
function reduceChatEvents(events) {
	const work = /* @__PURE__ */ new Map();
	const order = [];
	for (const event of events) switch (event.type) {
		case "turn-start":
			turnFor(work, order, event.turnId);
			break;
		case "text-start": {
			const { turn, textById } = turnFor(work, order, event.turnId);
			markErrorsRetried(turn);
			if (!textById.has(event.id)) {
				const item = {
					kind: "text",
					id: event.id,
					text: "",
					closed: false
				};
				textById.set(event.id, item);
				turn.items.push(item);
			}
			break;
		}
		case "text-delta": {
			const { turn, textById } = turnFor(work, order, event.turnId);
			markErrorsRetried(turn);
			let item = textById.get(event.id);
			if (!item) {
				item = {
					kind: "text",
					id: event.id,
					text: "",
					closed: false
				};
				textById.set(event.id, item);
				turn.items.push(item);
			}
			item.text += event.delta;
			break;
		}
		case "text-end": {
			const item = work.get(event.turnId)?.textById.get(event.id);
			if (item) item.closed = true;
			break;
		}
		case "tool-call-start": {
			const { turn, callById } = turnFor(work, order, event.turnId);
			markErrorsRetried(turn);
			if (!callById.has(event.callId)) {
				const call = {
					callId: event.callId,
					name: event.name,
					argsText: "",
					status: "building"
				};
				callById.set(event.callId, call);
				appendToolCall(turn, call);
			}
			break;
		}
		case "tool-call-delta": {
			const call = work.get(event.turnId)?.callById.get(event.callId);
			if (call) call.argsText += event.argsTextDelta;
			break;
		}
		case "tool-call-ready": {
			const { turn, callById } = turnFor(work, order, event.turnId);
			markErrorsRetried(turn);
			let call = callById.get(event.callId);
			if (!call) {
				call = {
					callId: event.callId,
					name: event.name,
					argsText: "",
					status: "building"
				};
				callById.set(event.callId, call);
				appendToolCall(turn, call);
			}
			call.name = event.name;
			call.args = event.args;
			call.status = "ready";
			break;
		}
		case "tool-result": {
			const call = work.get(event.turnId)?.callById.get(event.callId);
			if (call) {
				call.ok = event.ok;
				call.status = event.ok ? "ok" : "error";
				if (event.result !== void 0) call.result = event.result;
				if (event.error !== void 0) call.error = event.error;
			}
			break;
		}
		case "usage": {
			const { turn } = turnFor(work, order, event.turnId);
			turn.usage = {
				inputTokens: event.inputTokens,
				outputTokens: event.outputTokens
			};
			break;
		}
		case "abort": {
			const { turn } = turnFor(work, order, event.turnId);
			turn.status = "aborted";
			break;
		}
		case "turn-end": {
			const { turn } = turnFor(work, order, event.turnId);
			if (turn.stopReason !== void 0) break;
			turn.stopReason = event.stopReason;
			turn.status = event.stopReason === "aborted" ? "aborted" : "complete";
			break;
		}
		case "error": {
			const { turn } = turnFor(work, order, event.turnId ?? order[order.length - 1] ?? "");
			markErrorsRetried(turn);
			turn.items.push({
				kind: "error",
				code: event.code,
				message: event.message,
				retryable: event.retryable,
				superseded: false
			});
			break;
		}
		default: break;
	}
	return order.map((turnId) => work.get(turnId).turn);
}
//#endregion
//#region src/renderers/chat.ts
function isRecord(value) {
	return typeof value === "object" && value !== null;
}
function asString(value) {
	return typeof value === "string" ? value : "";
}
/**
* A friendly one-line summary of a tool call, derived from its `dashboard.*` method
* name + args (SPEC examples). Falls back to the raw method when no heuristic fits.
*/
function friendlyToolLabel(name, args) {
	const a = isRecord(args) ? args : {};
	switch (name.startsWith("dashboard.") ? name.slice(10) : name) {
		case "tab.create": {
			const label = asString(a.title) || asString(a.slug);
			return label ? t("dashboard.widget.chat.tool.createdTab", { name: label }) : name;
		}
		case "widget.add": {
			const id = asString(a.id) || asString(a.widgetId);
			return id ? t("dashboard.widget.chat.tool.addedWidget", { id }) : name;
		}
		case "workspace.get": return t("dashboard.widget.chat.tool.readBoard");
		default: return name;
	}
}
/** The "✓✓✗"-style per-call summary string for a tool group's chip. */
function toolGroupMarks(calls) {
	return calls.map((call) => {
		const mark = chatToolMark(call);
		return mark === "ok" ? "✓" : mark === "error" ? "✗" : "·";
	}).join("");
}
function toolActionsLabel(count) {
	return count === 1 ? t("dashboard.widget.chat.actionsOne") : t("dashboard.widget.chat.actionsMany", { count: String(count) });
}
/** Pretty-print an args/result payload for the expandable detail rows. */
function formatPayload(value) {
	try {
		return JSON.stringify(value, null, 2);
	} catch {
		return String(value);
	}
}
/** Render one tool call as a log row: a shimmer while building, else name + JSON details. */
function renderToolRow(call, aborted) {
	const building = (call.status === "building" || call.status === "ready") && !call.ok;
	if (building && !aborted) return b`<div class="dashboard-chat__tool-row dashboard-chat__tool-row--building">
      <span class="dashboard-chat__shimmer"></span>
      <span class="dashboard-chat__tool-name">${friendlyToolLabel(call.name, call.args)}</span>
      <span class="dashboard-chat__tool-note">${t("dashboard.widget.chat.building")}</span>
    </div>`;
	const mark = chatToolMark(call);
	const hasArgs = call.args !== void 0 || call.argsText.length > 0;
	const hasResult = call.result !== void 0 || call.error !== void 0;
	return b`<div
    class="dashboard-chat__tool-row"
    data-status=${aborted && building ? "cancelled" : mark}
  >
    <span class="dashboard-chat__tool-name">
      <span class="dashboard-chat__tool-mark" aria-hidden="true"
        >${mark === "ok" ? "✓" : mark === "error" ? "✗" : "·"}</span
      >
      ${friendlyToolLabel(call.name, call.args)}
    </span>
    ${hasArgs ? b`<details class="dashboard-chat__tool-detail">
            <summary>${t("dashboard.widget.chat.args")}</summary>
            <pre>${call.args !== void 0 ? formatPayload(call.args) : call.argsText}</pre>
          </details>` : A}
    ${hasResult ? b`<details class="dashboard-chat__tool-detail">
            <summary>${t("dashboard.widget.chat.result")}</summary>
            <pre>${formatPayload(call.error ?? call.result)}</pre>
          </details>` : A}
  </div>`;
}
/** Render a run of consecutive tool calls as one collapsed group chip. */
function renderToolGroup(group, aborted) {
	const count = group.calls.length;
	return b`<details class="dashboard-chat__tools" data-test-id="dashboard-chat-tools">
    <summary class="dashboard-chat__chip">
      <span aria-hidden="true">🔧</span>
      <span class="dashboard-chat__chip-count">${toolActionsLabel(count)}</span>
      <span class="dashboard-chat__chip-sep" aria-hidden="true">·</span>
      <span class="dashboard-chat__chip-marks">${toolGroupMarks(group.calls)}</span>
    </summary>
    <div class="dashboard-chat__tool-log">
      ${group.calls.map((call) => renderToolRow(call, aborted))}
    </div>
  </details>`;
}
/** Render one assistant turn: a role label plus its interleaved text/tool/error items. */
function renderAssistantTurn(turn) {
	const aborted = turn.status === "aborted";
	return b`<div
    class="dashboard-chat__turn dashboard-chat__turn--assistant"
    data-test-id="dashboard-chat-turn"
    data-status=${turn.status}
  >
    <div class="dashboard-chat__role">${t("dashboard.widget.chat.roleAssistant")}</div>
    ${turn.items.map((item) => {
		if (item.kind === "text") return b`<div class="dashboard-chat__text markdown-body">
          ${o(toSanitizedMarkdownHtml(item.text))}
        </div>`;
		if (item.kind === "tools") return renderToolGroup(item, aborted);
		return b`<div
        class="dashboard-chat__error"
        role="alert"
        data-test-id="dashboard-chat-error"
      >
        <span class="dashboard-chat__error-message">${item.message}</span>
        ${item.retryable && item.superseded ? b`<span class="dashboard-chat__error-retry"
                >${t("dashboard.widget.chat.retrying")}</span
              >` : A}
      </div>`;
	})}
  </div>`;
}
/** Render a user message bubble (plain text, left-aligned, role-labelled). */
function renderUserTurn(text) {
	return b`<div
    class="dashboard-chat__turn dashboard-chat__turn--user"
    data-test-id="dashboard-chat-user"
  >
    <div class="dashboard-chat__role">${t("dashboard.widget.chat.roleUser")}</div>
    <div class="dashboard-chat__text">${text}</div>
  </div>`;
}
/** The distance-from-bottom (px) within which the transcript sticks to the newest content. */
const STICK_TO_BOTTOM_PX = 100;
/**
* The per-widget interactive island. One instance per widget id (keyed in the
* module map below); it holds the event log + live subscription and re-renders its
* own subtree. The parent view feeds a fresh `ctx` on every doc change via
* `setContext`, keeping `registryPending` (the inline approval card) current.
*/
var ChatController = class {
	constructor(widgetId) {
		this.widgetId = widgetId;
		this.root = null;
		this.ctx = null;
		this.widget = null;
		this.events = [];
		this.unsubscribe = null;
		this.userMessages = /* @__PURE__ */ new Map();
		this.pendingUserText = null;
		this.sending = false;
		this.stickToBottom = true;
		this.rootRef = (element) => {
			if (element instanceof HTMLElement) this.mount(element);
			else this.destroy();
		};
		this.onSubmit = (event) => {
			event.preventDefault();
			this.send();
		};
		this.onTextareaKey = (event) => {
			if (event.key === "Enter" && !event.shiftKey) {
				event.preventDefault();
				this.send();
			}
		};
		this.onStop = (turnId) => {
			this.ctx?.chat?.abort(turnId).catch(() => {});
		};
		this.onScroll = (event) => {
			const el = event.currentTarget;
			this.stickToBottom = el.scrollHeight - el.scrollTop - el.clientHeight < STICK_TO_BOTTOM_PX;
			this.updateJumpPill();
		};
		this.jumpToLatest = () => {
			const scroll = this.root?.querySelector(".dashboard-chat__scroll");
			if (scroll) {
				this.stickToBottom = true;
				scroll.scrollTop = scroll.scrollHeight;
				this.updateJumpPill();
			}
		};
	}
	/** Store the latest render context/widget (parent re-render) and refresh the island. */
	setContext(ctx, widget) {
		this.ctx = ctx;
		this.widget = widget;
		if (this.root) this.renderIsland();
	}
	mount(element) {
		this.root = element;
		this.unsubscribe?.();
		this.unsubscribe = null;
		this.events = [];
		this.userMessages.clear();
		this.pendingUserText = null;
		this.sending = false;
		this.stickToBottom = true;
		this.renderIsland();
		const chat = this.ctx?.chat;
		if (!chat) return;
		chat.history().then((events) => {
			this.events = [...events, ...this.events];
			this.renderIsland();
		}).catch(() => {});
		this.unsubscribe = chat.subscribe((event) => {
			this.events.push(event);
			this.renderIsland();
		});
	}
	destroy() {
		this.unsubscribe?.();
		this.unsubscribe = null;
		this.root = null;
		controllers.delete(this.widgetId);
	}
	liveTurnId(turns) {
		for (let i = turns.length - 1; i >= 0; i -= 1) if (turns[i].status === "streaming") return turns[i].turnId;
	}
	send() {
		const chat = this.ctx?.chat;
		const textarea = this.root?.querySelector(".dashboard-chat__textarea");
		if (!chat || !textarea) return;
		const message = textarea.value.trim();
		if (!message || this.sending) return;
		textarea.value = "";
		this.pendingUserText = message;
		this.sending = true;
		this.stickToBottom = true;
		this.renderIsland();
		chat.send(message).then(({ turnId }) => {
			this.userMessages.set(turnId, message);
		}).catch(() => {}).finally(() => {
			this.pendingUserText = null;
			this.sending = false;
			this.renderIsland();
		});
	}
	updateJumpPill() {
		const pill = this.root?.querySelector(".dashboard-chat__jump");
		if (pill) pill.hidden = this.stickToBottom;
	}
	renderIsland() {
		if (!this.root) return;
		const turns = reduceChatEvents(this.events);
		const liveTurnId = this.liveTurnId(turns);
		const isLive = liveTurnId !== void 0 || this.sending;
		const pending = this.ctx?.registryPending ?? [];
		const canApprove = Boolean(this.ctx?.approveWidget);
		const showApprovals = isLive && canApprove && pending.length > 0;
		const empty = turns.length === 0 && this.pendingUserText === null;
		const disconnected = !this.ctx?.chat;
		D(b`
        <div class="dashboard-chat__scroll" @scroll=${this.onScroll}>
          ${empty ? b`<div class="dashboard-chat__empty" data-test-id="dashboard-chat-empty">
                  ${t("dashboard.widget.chat.empty")}
                </div>` : A}
          ${turns.map((turn) => {
			const userText = this.userMessages.get(turn.turnId);
			return b`${userText !== void 0 ? renderUserTurn(userText) : A}
            ${renderAssistantTurn(turn)}`;
		})}
          ${this.pendingUserText !== null ? renderUserTurn(this.pendingUserText) : A}
          ${showApprovals ? pending.map((name) => b`<div
                      class="dashboard-chat__approval"
                      data-test-id="dashboard-chat-approval"
                    >
                      <span class="dashboard-chat__approval-title"
                        >${t("dashboard.widget.chat.approveTitle", { name })}</span
                      >
                      <span class="dashboard-chat__approval-actions">
                        <button
                          class="bs-btn bs-btn--small bs-btn--primary"
                          type="button"
                          data-test-id="dashboard-chat-approve"
                          @click=${() => this.ctx?.approveWidget?.(name, "approved")}
                        >
                          ${t("dashboard.widget.chat.approve")}
                        </button>
                        <button
                          class="bs-btn bs-btn--small"
                          type="button"
                          data-test-id="dashboard-chat-reject"
                          @click=${() => this.ctx?.approveWidget?.(name, "rejected")}
                        >
                          ${t("dashboard.widget.chat.reject")}
                        </button>
                      </span>
                    </div>`) : A}
        </div>
        <button
          class="dashboard-chat__jump"
          type="button"
          hidden
          data-test-id="dashboard-chat-jump"
          @click=${this.jumpToLatest}
        >
          ${t("dashboard.widget.chat.jumpToLatest")} ↓
        </button>
        <form class="dashboard-chat__input" @submit=${this.onSubmit}>
          <textarea
            class="dashboard-chat__textarea"
            data-test-id="dashboard-chat-textarea"
            rows="2"
            ?disabled=${disconnected}
            placeholder=${this.placeholder()}
            @keydown=${this.onTextareaKey}
          ></textarea>
          <div class="dashboard-chat__input-actions">
            ${liveTurnId !== void 0 ? b`<button
                    class="bs-btn bs-btn--small dashboard-chat__stop"
                    type="button"
                    data-test-id="dashboard-chat-stop"
                    @click=${() => this.onStop(liveTurnId)}
                  >
                    ${t("dashboard.widget.chat.stop")}
                  </button>` : A}
            <button
              class="bs-btn bs-btn--small bs-btn--primary dashboard-chat__send"
              type="submit"
              data-test-id="dashboard-chat-send"
              ?disabled=${disconnected}
            >
              ${t("dashboard.widget.chat.send")}
            </button>
          </div>
        </form>
        ${disconnected ? b`<div class="dashboard-chat__hint" data-test-id="dashboard-chat-disconnected">
                ${t("dashboard.widget.chat.disconnected")}
              </div>` : A}
      `, this.root);
		if (this.stickToBottom) {
			const scroll = this.root.querySelector(".dashboard-chat__scroll");
			if (scroll) scroll.scrollTop = scroll.scrollHeight;
		}
		this.updateJumpPill();
	}
	placeholder() {
		return asString((isRecord(this.widget?.props) ? this.widget.props : {}).placeholder) || t("dashboard.widget.chat.placeholder");
	}
};
/** One live controller per widget id. Created lazily; removed on the widget's unmount. */
const controllers = /* @__PURE__ */ new Map();
/**
* Renders builtin:chat. The renderer stays a pure function returning the island's
* container; the `ChatController` (keyed by widget id) owns the interactive state
* and its own render loop, hydrated via the `ref` callback (the `notes` pattern).
*/
function renderChat(widget, _value, ctx) {
	let controller = controllers.get(widget.id);
	if (!controller) {
		controller = new ChatController(widget.id);
		controllers.set(widget.id, controller);
	}
	controller.setContext(ctx, widget);
	return b`<div
    class="dashboard-chat"
    data-test-id="dashboard-chat"
    ${n(controller.rootRef)}
  ></div>`;
}
//#endregion
//#region src/renderers/cron.ts
function statusClass(status) {
	if (status === "ok") return "dashboard-badge--ok";
	if (status === "error") return "dashboard-badge--error";
	return "dashboard-badge--muted";
}
function renderCron(widget, value) {
	const model = mapCron(widget, value);
	if (model.jobs.length === 0) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.cron.empty")}
    </div>`;
	return b`
    <ul class="dashboard-list dashboard-cron" data-test-id="dashboard-cron">
      ${model.jobs.map((job) => b`
          <li class="dashboard-list__row ${job.enabled ? "" : "dashboard-list__row--disabled"}">
            <span class="dashboard-list__label">${job.name}</span>
            <span class="dashboard-list__meta">
              ${job.nextRunAtMs !== null ? t("dashboard.widget.cron.next", { time: formatDateTimeMs(job.nextRunAtMs) }) : t("dashboard.widget.cron.noNext")}
            </span>
            ${job.lastStatus ? b`<span class="dashboard-badge ${statusClass(job.lastStatus)}"
                    >${job.lastStatus}</span
                  >` : A}
          </li>
        `)}
    </ul>
  `;
}
//#endregion
//#region src/renderers/iframe-embed.ts
/** The iframe `sandbox` attribute for an embed mode. `strict` grants nothing. */
function resolveEmbedSandbox(mode) {
	return mode === "scripts" ? "allow-scripts" : "";
}
function renderIframeEmbed(widget, _value, ctx) {
	const decision = evaluateEmbedUrl(widgetProps(widget).url, { allowExternalEmbedUrls: ctx.embed.allowExternalEmbedUrls });
	if (decision.status === "missing") return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.embed.missing")}
    </div>`;
	if (decision.status === "blocked") return b`<div class="dashboard-widget__placeholder" data-test-id="dashboard-embed-blocked">
      ${decision.reason === "external" ? t("dashboard.widget.embed.blockedExternal") : t("dashboard.widget.embed.blockedScheme")}
    </div>`;
	return b`<iframe
    class="dashboard-embed__frame"
    data-test-id="dashboard-embed-frame"
    src=${decision.url}
    title=${widget.title}
    sandbox=${resolveEmbedSandbox(ctx.embed.embedSandboxMode)}
    referrerpolicy="no-referrer"
    loading="lazy"
  ></iframe>`;
}
//#endregion
//#region src/renderers/instances.ts
function renderInstances(widget, value) {
	const model = mapInstances(widget, value);
	if (model.instances.length === 0) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.instances.empty")}
    </div>`;
	return b`
    <ul class="dashboard-list dashboard-instances" data-test-id="dashboard-instances">
      ${model.instances.map((instance) => b`
          <li class="dashboard-list__row">
            <span
              class="dashboard-dot ${instance.healthy ? "dashboard-dot--ok" : "dashboard-dot--warn"}"
              aria-hidden="true"
            ></span>
            <span class="dashboard-list__label">${instance.id}</span>
            ${instance.detail ? b`<span class="dashboard-list__meta">${instance.detail}</span>` : A}
            ${instance.lastInputMs !== null ? b`<span class="dashboard-list__meta"
                    >${t("dashboard.widget.instances.idle", { duration: formatMs(instance.lastInputMs) })}</span
                  >` : A}
          </li>
        `)}
    </ul>
  `;
}
//#endregion
//#region src/renderers/markdown.ts
function renderMarkdown(widget, value) {
	const source = mapMarkdownSource(widget, value);
	if (!source.trim()) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.markdownEmpty")}
    </div>`;
	return b`<div class="dashboard-markdown markdown-body">
    ${o(toSanitizedMarkdownHtml(source))}
  </div>`;
}
//#endregion
//#region src/renderers/notes.ts
/** Seed text from `props` when there is no persisted state yet (author-provided default). */
function notesSeedText(widget) {
	const props = widgetProps(widget);
	if (typeof props.text === "string") return props.text;
	return "";
}
/**
* Callback ref that hydrates the textarea from the widget's persisted state, then
* wires debounced persistence on input. The `ref` directive calls this with the
* element on connect (and `undefined` on disconnect). All state errors are
* swallowed: a failed load/save leaves the pad usable rather than throwing into
* the cell's error boundary.
*/
function bindNotesEditor(state) {
	return (element) => {
		if (!(element instanceof HTMLTextAreaElement)) return;
		const textarea = element;
		if (textarea.dataset.notesBound === "1") return;
		textarea.dataset.notesBound = "1";
		state.get().then((result) => {
			if (textarea.dataset.notesDirty !== "1") textarea.value = notesTextFromState(result.state);
		}).catch(() => {});
		let timer;
		textarea.addEventListener("input", () => {
			textarea.dataset.notesDirty = "1";
			const next = textarea.value;
			if (timer !== void 0) clearTimeout(timer);
			timer = setTimeout(() => {
				state.set(next).catch(() => {});
			}, 500);
		});
	};
}
function renderNotes(widget, _value, ctx) {
	const placeholder = t("dashboard.widget.notes.placeholder");
	if (!ctx.state) {
		const seed = notesSeedText(widget);
		return b`
      <div class="dashboard-notes dashboard-notes--readonly" data-test-id="dashboard-notes">
        <textarea
          class="dashboard-notes__pad"
          data-test-id="dashboard-notes-pad"
          readonly
          aria-label=${widget.title}
          placeholder=${placeholder}
        >
${seed}</textarea>
        <div class="dashboard-notes__hint" data-test-id="dashboard-notes-hint">
          ${t("dashboard.widget.notes.readonlyHint")}
        </div>
      </div>
    `;
	}
	return b`
    <div class="dashboard-notes" data-test-id="dashboard-notes">
      <textarea
        class="dashboard-notes__pad"
        data-test-id="dashboard-notes-pad"
        aria-label=${widget.title}
        placeholder=${placeholder}
        ${n(bindNotesEditor(ctx.state))}
      ></textarea>
    </div>
  `;
}
//#endregion
//#region src/renderers/preview.ts
const PREVIEW_VIEWPORTS = [
	"desktop",
	"tablet",
	"mobile"
];
function viewportClass(viewport) {
	return `dashboard-preview__frame-wrap dashboard-preview__frame-wrap--${viewport}`;
}
function renderPreview(widget, _value, ctx) {
	const decision = evaluateEmbedUrl(widgetProps(widget).url, { allowExternalEmbedUrls: ctx.embed.allowExternalEmbedUrls });
	if (decision.status === "missing") return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.preview.missing")}
    </div>`;
	if (decision.status === "blocked") return b`<div class="dashboard-widget__placeholder" data-test-id="dashboard-preview-blocked">
      ${decision.reason === "external" ? t("dashboard.widget.preview.blockedExternal") : t("dashboard.widget.preview.blockedScheme")}
    </div>`;
	const initialViewport = mapPreviewViewport(widget);
	const frameRef = e$1();
	const wrapRef = e$1();
	const reload = () => {
		const frame = frameRef.value;
		if (frame) {
			const src = frame.getAttribute("src");
			if (src !== null) frame.setAttribute("src", src);
		}
	};
	const setViewport = (viewport) => {
		const wrap = wrapRef.value;
		if (wrap) wrap.className = viewportClass(viewport);
	};
	return b`<div class="dashboard-preview">
    <div class="dashboard-preview__toolbar" role="toolbar">
      <div class="dashboard-preview__viewports" role="group">
        ${PREVIEW_VIEWPORTS.map((viewport) => b`<button
              class="dashboard-preview__viewport"
              type="button"
              data-test-id=${`dashboard-preview-viewport-${viewport}`}
              data-viewport=${viewport}
              title=${t(`dashboard.widget.preview.viewport.${viewport}`)}
              aria-label=${t(`dashboard.widget.preview.viewport.${viewport}`)}
              @click=${() => setViewport(viewport)}
            >
              ${t(`dashboard.widget.preview.viewport.${viewport}`)}
            </button>`)}
      </div>
      <button
        class="dashboard-preview__reload"
        type="button"
        data-test-id="dashboard-preview-reload"
        title=${t("dashboard.widget.preview.reload")}
        aria-label=${t("dashboard.widget.preview.reload")}
        @click=${reload}
      >
        ${t("dashboard.widget.preview.reload")}
      </button>
    </div>
    <div class=${viewportClass(initialViewport)} ${n(wrapRef)}>
      <iframe
        class="dashboard-embed__frame dashboard-preview__frame"
        data-test-id="dashboard-preview-frame"
        ${n(frameRef)}
        src=${decision.url}
        title=${widget.title}
        sandbox=${resolveEmbedSandbox(ctx.embed.embedSandboxMode)}
        referrerpolicy="no-referrer"
        loading="lazy"
      ></iframe>
    </div>
  </div>`;
}
//#endregion
//#region src/renderers/sessions.ts
function renderSessions(widget, value, ctx) {
	const model = mapSessions(widget, value);
	if (model.rows.length === 0) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.sessions.empty")}
    </div>`;
	const href = (key) => ctx?.sessionHref?.(key) ?? "#";
	const onNavigate = ctx?.onNavigate;
	return b`
    <ul class="dashboard-list dashboard-sessions" data-test-id="dashboard-sessions">
      ${model.rows.map((row) => b`
          <li class="dashboard-list__row">
            <a
              class="dashboard-list__link"
              href=${href(row.key)}
              @click=${onNavigate ? (event) => {
		event.preventDefault();
		onNavigate(row.key);
	} : A}
            >
              <span
                class="dashboard-dot ${row.active ? "dashboard-dot--live" : ""}"
                aria-hidden="true"
              ></span>
              <span class="dashboard-list__label">${row.label}</span>
              ${row.updatedAt !== null ? b`<span class="dashboard-list__meta"
                      >${formatDateTimeMs(row.updatedAt)}</span
                    >` : A}
            </a>
          </li>
        `)}
    </ul>
  `;
}
//#endregion
//#region src/renderers/stat-card.ts
function renderStatCard(widget, value) {
	const model = mapStatCard(widget, value);
	return b`
    <div class="dashboard-stat">
      <div class="dashboard-stat__value">${model.display ?? t("dashboard.widget.stat.empty")}</div>
      ${model.label ? b`<div class="dashboard-stat__label">${model.label}</div>` : A}
    </div>
  `;
}
//#endregion
//#region src/renderers/table.ts
function renderCell(value) {
	if (value === null || value === void 0) return "";
	if (typeof value === "string") return value;
	if (typeof value === "number" || typeof value === "boolean") return String(value);
	return JSON.stringify(value);
}
function renderTable(widget, value) {
	const model = mapTable(widget, value);
	if (model.total === 0 || model.columns.length === 0) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.table.empty")}
    </div>`;
	const remaining = model.total - model.shown;
	return b`
    <div class="dashboard-table">
      <table class="dashboard-table__grid">
        <thead>
          <tr>
            ${model.columns.map((column) => b`<th scope="col">${column}</th>`)}
          </tr>
        </thead>
        <tbody>
          ${model.rows.map((row) => b`
              <tr>
                ${model.columns.map((column) => b`<td>${renderCell(row[column])}</td>`)}
              </tr>
            `)}
        </tbody>
      </table>
      ${remaining > 0 ? b`<div class="dashboard-table__footer">
              ${t("dashboard.widget.table.more", { count: String(remaining) })}
            </div>` : A}
    </div>
  `;
}
//#endregion
//#region src/renderers/usage.ts
function renderUsage(widget, value) {
	const model = mapUsage(widget, value);
	return b`
    <div class="dashboard-usage" data-test-id="dashboard-usage">
      <div class="dashboard-usage__metric">
        <div class="dashboard-usage__value">${formatCost(model.cost)}</div>
        <div class="dashboard-usage__label">${t("dashboard.widget.usage.cost")}</div>
      </div>
      <div class="dashboard-usage__metric">
        <div class="dashboard-usage__value">${formatTokens(model.tokens)}</div>
        <div class="dashboard-usage__label">${t("dashboard.widget.usage.tokens")}</div>
      </div>
    </div>
  `;
}
//#endregion
//#region src/renderers/index.ts
const BUILTIN_WIDGET_RENDERERS = {
	"stat-card": (widget, value) => renderStatCard(widget, value),
	markdown: (widget, value) => renderMarkdown(widget, value),
	table: (widget, value) => renderTable(widget, value),
	"iframe-embed": renderIframeEmbed,
	preview: renderPreview,
	sessions: (widget, value, ctx) => renderSessions(widget, value, ctx),
	usage: (widget, value) => renderUsage(widget, value),
	cron: (widget, value) => renderCron(widget, value),
	instances: (widget, value) => renderInstances(widget, value),
	activity: (widget, value) => renderActivity(widget, value),
	chart: (widget, value) => renderChart(widget, value),
	notes: renderNotes,
	"action-form": renderActionForm,
	"action-button": renderActionButton,
	"agent-status": (widget, value) => renderAgentStatus(widget, value),
	approvals: renderApprovals,
	chat: renderChat
};
function getBuiltinRenderer(kind) {
	const name = kind.startsWith("builtin:") ? kind.slice(8) : kind;
	return BUILTIN_WIDGET_RENDERERS[name];
}
//#endregion
//#region src/boardstate-widget-cell.ts
/**
* Visible widget title with a trailing " (custom)" provenance suffix stripped: the
* suffix is redundant with the AI/provenance chip and only causes truncation; the
* full title is still exposed via the `title=` attribute.
*/
function displayWidgetTitle(title) {
	return title.replace(/\s*\(custom\)\s*$/iu, "").trim() || title;
}
/**
* Renders the provenance chip when a widget was authored by an agent. On a MULTI-AGENT
* board the view passes an `agentChip` and this renders the per-agent COLOURED chip
* (short id, full actor on hover, deterministic hue) — the distinguishing affordance of
* SPEC §17.3 (#59). Otherwise it falls back to the plain "AI" chip.
*/
function renderProvenanceChip(widget, agentChip) {
	const agentId = dashboardAgentProvenance(widget.createdBy);
	if (!agentId) return A;
	if (agentChip) return b`<span
      class=${agentChip.dimmed ? "dashboard-widget__agent dashboard-widget__agent--dimmed" : "dashboard-widget__agent"}
      style="--dashboard-agent-hue: ${agentChip.hue}"
      data-test-id="dashboard-widget-agent-chip"
      data-agent=${agentChip.actor}
      title=${t("dashboard.widget.agentChipTooltip", { agent: agentChip.actor })}
      >${agentChip.short}</span
    >`;
	return b`<span
    class="dashboard-widget__provenance"
    title=${t("dashboard.widget.provenanceTooltip", { agent: agentId })}
    >${t("dashboard.widget.provenanceChip")}</span
  >`;
}
/** Subtle badge marking a temporary (ephemeral) Living Answer; pinning clears it. */
function renderEphemeralBadge(widget) {
	if (!widget.ephemeral) return A;
	return b`<span
    class="dashboard-widget__ephemeral"
    data-test-id="dashboard-widget-ephemeral"
    title=${t("dashboard.widget.ephemeralTooltip")}
    >${t("dashboard.widget.ephemeralBadge")}</span
  >`;
}
/**
* Blame line shown at the top of the cell menu (M2): "Created by {actor} · v{n}",
* with a logbook deep link when the author is an agent and the link is derivable.
* When the logbook seam yields no link, the provenance line renders on its own.
*/
function renderBlame(blame) {
	return b`
    <div class="dashboard-widget__blame" role="note" data-test-id="dashboard-widget-blame">
      <span class="dashboard-widget__blame-text">${blame.firstSeenVersion !== void 0 ? t("dashboard.widget.blame.createdByVersion", {
		actor: blame.actor,
		version: String(blame.firstSeenVersion)
	}) : t("dashboard.widget.blame.createdBy", { actor: blame.actor })}</span>
      ${blame.agentId !== null && Boolean(blame.logbookHref) ? b`<a
              class="dashboard-widget__blame-link"
              href=${blame.logbookHref}
              target="_blank"
              rel="noopener noreferrer"
              data-test-id="dashboard-widget-blame-link"
              >${icons.externalLink} ${t("dashboard.widget.blame.logbookLink")}</a
            >` : A}
    </div>
  `;
}
function renderMenu(widget, callbacks, blame) {
	return b`
    <div class="dashboard-widget__menu" role="menu">
      ${blame ? renderBlame(blame) : A}
      ${widget.ephemeral ? b`<button
              class="dashboard-widget__menu-item"
              type="button"
              role="menuitem"
              data-test-id="dashboard-widget-pin"
              @click=${() => callbacks.onPin(widget)}
            >
              ${t("dashboard.widget.menu.pin")}
            </button>` : A}
      <button
        class="dashboard-widget__menu-item"
        type="button"
        role="menuitem"
        @click=${() => callbacks.onEditTitle(widget)}
      >
        ${t("dashboard.widget.menu.editTitle")}
      </button>
      <button
        class="dashboard-widget__menu-item"
        type="button"
        role="menuitem"
        @click=${() => callbacks.onMoveToTab(widget)}
      >
        ${t("dashboard.widget.menu.moveToTab")}
      </button>
      <button
        class="dashboard-widget__menu-item"
        type="button"
        role="menuitem"
        @click=${() => callbacks.onHide(widget)}
      >
        ${t("dashboard.widget.menu.hide")}
      </button>
      <button
        class="dashboard-widget__menu-item dashboard-widget__menu-item--danger"
        type="button"
        role="menuitem"
        @click=${() => callbacks.onRemove(widget)}
      >
        ${t("dashboard.widget.menu.remove")}
      </button>
    </div>
  `;
}
/**
* Renders a builtin widget body via the renderer registry. A binding error is
* re-thrown so the cell error boundary shows it inline; unknown/custom kinds render
* a placeholder (custom widgets are dispatched by renderWidgetBody first).
*/
function renderBuiltinWidget(widget, binding, ctx) {
	if (binding && "error" in binding) throw new Error(binding.error);
	const value = binding && "value" in binding ? binding.value : void 0;
	const renderer = getBuiltinRenderer(widget.kind);
	if (renderer) return renderer(widget, value, ctx);
	if (widget.kind.startsWith("custom:")) return b`<div class="dashboard-widget__placeholder">
      ${t("dashboard.widget.customPlaceholder")}
    </div>`;
	return b`<div class="dashboard-widget__placeholder">
    ${t("dashboard.widget.unknownKind", { kind: widget.kind })}
  </div>`;
}
/**
* Renders a `custom:<name>` widget. The registry status is the render gate,
* mirroring the server's approved-only serving gate:
* - `approved` → the sandboxed iframe host (only path that ever builds an iframe).
* - `pending`  → a placeholder card with operator-only Approve/Reject.
* - `rejected` / unknown → a neutral placeholder; NO iframe is constructed.
*/
function renderCustomWidget(widget, custom) {
	if (custom.status === "approved") {
		if (!custom.manifest) return b`<div
        class="dashboard-widget__placeholder"
        data-test-id="dashboard-custom-loading"
      >
        ${t("dashboard.widget.customLoading")}
      </div>`;
		return renderCustomWidgetHost({
			widget,
			manifest: custom.manifest,
			context: custom.host
		});
	}
	if (custom.status === "pending") {
		const author = dashboardAgentProvenance(widget.createdBy);
		return b`
      <div
        class="dashboard-widget__approval"
        role="group"
        data-test-id="dashboard-custom-pending"
        aria-label=${t("dashboard.widget.approval.title")}
      >
        <div class="dashboard-widget__approval-title">${t("dashboard.widget.approval.title")}</div>
        <div class="dashboard-widget__approval-sub">
          ${author ? t("dashboard.widget.approval.byAgent", { agent: author }) : t("dashboard.widget.approval.byUnknown")}
        </div>
        <div class="dashboard-widget__approval-actions">
          <button
            class="bs-btn bs-btn--small bs-btn--primary"
            type="button"
            data-test-id="dashboard-custom-approve"
            @click=${() => custom.onApprove(widget)}
          >
            ${t("dashboard.widget.approval.approve")}
          </button>
          <button
            class="bs-btn bs-btn--small"
            type="button"
            data-test-id="dashboard-custom-reject"
            @click=${() => custom.onReject(widget)}
          >
            ${t("dashboard.widget.approval.reject")}
          </button>
        </div>
      </div>
    `;
	}
	return b`<div class="dashboard-widget__placeholder" data-test-id="dashboard-custom-rejected">
    ${t("dashboard.widget.approval.unavailable")}
  </div>`;
}
/**
* Error boundary around the widget body. Any throw during the builtin render (a
* broken widget, a bad binding) is caught and rendered as an error card in THIS
* cell — siblings and the shell keep rendering.
*/
function renderWidgetBody(widget, binding, ctx, callbacks, custom) {
	try {
		if (widget.kind.startsWith("custom:") && custom) return renderCustomWidget(widget, custom);
		return renderBuiltinWidget(widget, binding, ctx);
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		return b`
      <div class="dashboard-widget__error" role="alert" data-test-id="dashboard-widget-error">
        <div class="dashboard-widget__error-title">${t("dashboard.widget.errorTitle")}</div>
        <div class="dashboard-widget__error-humane">${t("dashboard.widget.errorHumane")}</div>
        <details class="dashboard-widget__error-detail">
          <summary>${t("dashboard.widget.errorDetailSummary")}</summary>
          <div class="dashboard-widget__error-message">${message}</div>
        </details>
        <button
          class="bs-btn bs-btn--small"
          type="button"
          @click=${() => callbacks.onRemove(widget)}
        >
          ${t("dashboard.widget.menu.remove")}
        </button>
      </div>
    `;
	}
}
function renderWidgetCell(props) {
	const { widget, callbacks } = props;
	const classes = [
		"dashboard-widget",
		widget.collapsed ? "dashboard-widget--collapsed" : "",
		props.pending ? "dashboard-widget--pending" : "",
		props.dragging ? "dashboard-widget--dragging" : "",
		props.dragging && props.dragTransform ? "dashboard-widget--carried" : "",
		props.agentChip?.dimmed ? "dashboard-widget--agent-dimmed" : ""
	].filter(Boolean).join(" ");
	const placement = gridPlacementStyle(widget.grid);
	return b`
    <section
      class=${classes}
      style=${props.dragging && props.dragTransform ? `${placement}; transform: ${props.dragTransform}` : placement}
      data-widget-id=${widget.id}
      data-test-id="dashboard-widget"
    >
      <header
        class="dashboard-widget__bar"
        @pointerdown=${(event) => callbacks.onMovePointerDown(widget, event)}
      >
        <button
          class="dashboard-widget__collapse"
          type="button"
          aria-expanded=${widget.collapsed ? "false" : "true"}
          aria-label=${widget.collapsed ? t("dashboard.widget.expand") : t("dashboard.widget.collapse")}
          @pointerdown=${(event) => event.stopPropagation()}
          @click=${() => callbacks.onToggleCollapse(widget)}
        >
          ${widget.collapsed ? icons.chevronRight : icons.chevronDown}
        </button>
        <span class="dashboard-widget__title" title=${widget.title}
          >${displayWidgetTitle(widget.title)}</span
        >
        ${renderProvenanceChip(widget, props.agentChip)} ${renderEphemeralBadge(widget)}
        <span
          class="dashboard-widget__handle"
          role="button"
          tabindex="0"
          aria-label=${t("dashboard.widget.moveHandle")}
          @keydown=${(event) => handleNudgeKey(event, widget, "move", callbacks)}
          >${icons.arrowUpDown}</span
        >
        <button
          class="dashboard-widget__menu-toggle"
          type="button"
          aria-haspopup="menu"
          aria-expanded=${props.menuOpen ? "true" : "false"}
          aria-label=${t("dashboard.widget.menuLabel")}
          @pointerdown=${(event) => event.stopPropagation()}
          @click=${() => callbacks.onToggleMenu(widget)}
        >
          ${icons.moreHorizontal}
        </button>
        ${props.menuOpen ? renderMenu(widget, callbacks, props.blame) : A}
      </header>
      ${widget.collapsed ? A : b`
              <div class="dashboard-widget__body">
                ${renderWidgetBody(widget, props.binding, props.builtinContext, callbacks, props.custom)}
              </div>
              <span
                class="dashboard-widget__resize"
                role="button"
                tabindex="0"
                aria-label=${t("dashboard.widget.resizeHandle")}
                @pointerdown=${(event) => callbacks.onResizePointerDown(widget, event)}
                @keydown=${(event) => handleNudgeKey(event, widget, "resize", callbacks)}
              ></span>
            `}
    </section>
  `;
}
/** Keyboard fallback for move/resize (a11y): arrow keys nudge by one grid unit. */
function handleNudgeKey(event, widget, mode, callbacks) {
	const direction = event.key === "ArrowLeft" ? "left" : event.key === "ArrowRight" ? "right" : event.key === "ArrowUp" ? "up" : event.key === "ArrowDown" ? "down" : null;
	if (!direction) return;
	event.preventDefault();
	callbacks.onKeyboardNudge(widget, mode, direction);
}
//#endregion
//#region src/agent-provenance.ts
const SHORT_ID_MAX = 10;
/**
* A stable 32-bit FNV-1a hash of a string → a hue in [0, 360). Deterministic and
* reload-stable (no `Math.random`, no insertion order), so an agent keeps one colour
* everywhere it appears. Not cryptographic — only a spread for visual distinction.
*/
function agentHue(actor) {
	let hash = 2166136261;
	for (let i = 0; i < actor.length; i++) {
		hash ^= actor.charCodeAt(i);
		hash = Math.imul(hash, 16777619);
	}
	return (hash >>> 0) % 360;
}
/** Cap an agent id to a header-safe length, ellipsizing when it overflows. */
function shortAgentId(agentId) {
	return agentId.length <= SHORT_ID_MAX ? agentId : `${agentId.slice(0, SHORT_ID_MAX - 1)}…`;
}
/**
* The distinct agent actors that authored a widget anywhere on the board, in stable
* (sorted) order. `user`/`system`-authored and unstamped widgets contribute nothing —
* only `agent:<id>` provenance counts, since scoping + chips are per-AGENT.
*/
function distinctAgentActors(workspace) {
	const actors = /* @__PURE__ */ new Set();
	for (const tab of workspace.tabs) for (const widget of tab.widgets) {
		const actor = widget.createdBy;
		if (actor && dashboardAgentProvenance(actor)) actors.add(actor);
	}
	return [...actors].sort();
}
/**
* Build the chip model for a widget's `createdBy`, or `null` when there is no agent chip
* to show (the widget is not agent-authored). `highlightedAgent` (the active filter, or
* null) drives `dimmed`.
*/
function agentChipFor(actor, highlightedAgent) {
	const agentId = dashboardAgentProvenance(actor);
	if (!agentId) return null;
	return {
		actor,
		agentId,
		short: shortAgentId(agentId),
		hue: agentHue(actor),
		dimmed: highlightedAgent !== null && actor !== highlightedAgent
	};
}
//#endregion
//#region src/boardstate-view.ts
const DEFAULT_EMBED = {
	embedSandboxMode: "strict",
	allowExternalEmbedUrls: false
};
function embedContext(policy) {
	if (!policy) return DEFAULT_EMBED;
	return {
		embedSandboxMode: policy.sandboxMode,
		allowExternalEmbedUrls: policy.allowExternalUrls
	};
}
function initialHistoryViewState() {
	return {
		open: false,
		loading: false,
		error: null,
		entries: [],
		snapshots: /* @__PURE__ */ new Map(),
		selectedVersion: null,
		confirmRestore: false,
		restoring: false
	};
}
/** Storage key remembering the operator's last registry index URL (w3). */
const GALLERY_URL_KEY = "boardstate:gallery-url:v1";
function readGalleryUrl(storage) {
	try {
		return storage?.getItem(GALLERY_URL_KEY) ?? "";
	} catch {
		return "";
	}
}
function persistGalleryUrl(storage, url) {
	try {
		storage?.setItem(GALLERY_URL_KEY, url);
	} catch {}
}
/** Shape one gallery fetch/install error into a display string. */
function formatGalleryError(err) {
	return err instanceof Error && err.message.trim() ? err.message.trim() : "Widget gallery error.";
}
/** localStorage flag so the first-visit onboarding banner stays dismissed across reloads. */
const ONBOARDING_DISMISS_KEY = "boardstate:onboarding-dismissed:v1";
function isOnboardingDismissed(storage) {
	try {
		return storage?.getItem(ONBOARDING_DISMISS_KEY) === "1";
	} catch {
		return false;
	}
}
function persistOnboardingDismissed(storage) {
	try {
		storage?.setItem(ONBOARDING_DISMISS_KEY, "1");
	} catch {}
}
const dashboardViewStates = /* @__PURE__ */ new WeakMap();
const dashboardMenuDismiss = /* @__PURE__ */ new WeakMap();
function teardownMenuDismiss(host) {
	const binding = dashboardMenuDismiss.get(host);
	if (!binding) return;
	document.removeEventListener("pointerdown", binding.onPointerDown, true);
	document.removeEventListener("keydown", binding.onKeyDown, true);
	dashboardMenuDismiss.delete(host);
}
function syncMenuDismiss(host, viewState, requestUpdate) {
	const menuOpen = viewState.openMenuWidgetId !== null;
	if (menuOpen === dashboardMenuDismiss.has(host)) return;
	if (!menuOpen) {
		teardownMenuDismiss(host);
		return;
	}
	const close = () => {
		if (viewState.openMenuWidgetId === null) return;
		viewState.openMenuWidgetId = null;
		teardownMenuDismiss(host);
		requestUpdate();
	};
	const onPointerDown = (event) => {
		const target = event.target;
		if (target instanceof Element && target.closest(".dashboard-widget__menu, .dashboard-widget__menu-toggle")) return;
		close();
	};
	const onKeyDown = (event) => {
		if (event.key === "Escape") {
			event.preventDefault();
			close();
		}
	};
	document.addEventListener("pointerdown", onPointerDown, true);
	document.addEventListener("keydown", onKeyDown, true);
	dashboardMenuDismiss.set(host, {
		onPointerDown,
		onKeyDown
	});
}
/** View-level teardown: drop menu-dismiss listeners + live stream subscriptions. */
function stopBoardstateView(host) {
	teardownMenuDismiss(host);
	teardownStreamSubscriptions(host);
}
/** Unsubscribe every live `stream` binding for `host` (tab-leave / disconnect / stop). */
function teardownStreamSubscriptions(host) {
	const viewState = dashboardViewStates.get(host);
	if (!viewState) return;
	for (const sub of viewState.streamSubs.values()) sub.unsubscribe();
	viewState.streamSubs.clear();
}
function getViewState(host, storage) {
	let state = dashboardViewStates.get(host);
	if (!state) {
		state = {
			openMenuWidgetId: null,
			drag: null,
			bindingResults: /* @__PURE__ */ new Map(),
			bindingLoads: /* @__PURE__ */ new Set(),
			bindingVersion: -1,
			streamSubs: /* @__PURE__ */ new Map(),
			streamValues: /* @__PURE__ */ new Map(),
			manifestCache: /* @__PURE__ */ new Map(),
			manifestLoads: /* @__PURE__ */ new Set(),
			dataVersion: 0,
			dialog: null,
			onboardingDismissed: isOnboardingDismissed(storage),
			collapsedTabGroups: /* @__PURE__ */ new Set(),
			lastPresenceSlug: null,
			history: initialHistoryViewState(),
			gallery: null,
			highlightedAgent: null
		};
		dashboardViewStates.set(host, state);
	}
	return state;
}
/** Read the current data-refresh counter for a host (used by the poll timer). */
function boardstateDataVersion(host) {
	return dashboardViewStates.get(host)?.dataVersion ?? 0;
}
/** Advance the data-refresh counter so the next render re-resolves bindings. */
function bumpBoardstateDataVersion(host) {
	const state = dashboardViewStates.get(host);
	if (state) state.dataVersion += 1;
}
/** Primary binding for a widget (first declared), if any. */
function primaryBinding(widget) {
	const bindings = widget.bindings;
	if (!bindings) return null;
	return Object.values(bindings)[0] ?? null;
}
/**
* Cache key mixing the workspace version with the data-refresh counter: a doc
* change OR a poll tick both invalidate resolved bindings. Overflow-safe.
*/
function bindingCacheKey(workspace, viewState) {
	return workspace.workspaceVersion * 1000003 + viewState.dataVersion;
}
/**
* Reconcile live `stream`-binding subscriptions against the active tab's widgets.
* Keyed by `workspaceVersion` so a poll tick never churns subscriptions; a doc
* change (or an event/pointer change) re-subscribes. A null transport (disconnect)
* tears every subscription down. Each pushed value lands in both `streamValues`
* (survives poll-tick cache clears) and `bindingResults` (the render cache).
*/
function reconcileStreamSubscriptions(viewState, transport, workspace, tab, requestUpdate) {
	if (!transport) {
		for (const sub of viewState.streamSubs.values()) sub.unsubscribe();
		viewState.streamSubs.clear();
		return;
	}
	const wanted = /* @__PURE__ */ new Map();
	for (const widget of tab.widgets) {
		const binding = primaryBinding(widget);
		if (binding?.source === "stream" && binding.event) wanted.set(widget.id, binding);
	}
	for (const [widgetId, sub] of viewState.streamSubs) {
		const binding = wanted.get(widgetId);
		if (!binding || sub.workspaceVersion !== workspace.workspaceVersion || sub.event !== binding.event || sub.pointer !== binding.pointer) {
			sub.unsubscribe();
			viewState.streamSubs.delete(widgetId);
			viewState.streamValues.delete(widgetId);
		}
	}
	for (const [widgetId, binding] of wanted) {
		if (viewState.streamSubs.has(widgetId)) continue;
		const unsubscribe = subscribeToStreamBinding(transport, binding, (result) => {
			viewState.streamValues.set(widgetId, result);
			viewState.bindingResults.set(widgetId, result);
			requestUpdate?.();
		});
		viewState.streamSubs.set(widgetId, {
			workspaceVersion: workspace.workspaceVersion,
			event: binding.event,
			...binding.pointer !== void 0 ? { pointer: binding.pointer } : {},
			unsubscribe
		});
	}
}
/**
* Resolve a `computed` primary binding from its sibling `inputs`: resolve each
* named input (leaf bindings only — the schema forbids computed→computed) then
* derive the value via the whitelisted op. Stream inputs are not one-shot
* resolvable and surface an error (computed reads settled values, not live pushes).
*/
async function resolveComputedForWidget(transport, widget, binding) {
	const siblings = widget.bindings ?? {};
	const values = [];
	for (const name of binding.inputs ?? []) {
		const input = siblings[name];
		if (!input) return { error: `Computed input not found: ${name}` };
		const result = await resolveBinding(transport, input);
		if ("error" in result) return { error: result.error };
		values.push(result.value);
	}
	return resolveComputedBinding(binding.op ?? "", values, binding.arg);
}
/** Kick off binding resolution for widgets on the active tab; cache per version. */
function ensureBindings(viewState, transport, workspace, tab, requestUpdate) {
	const key = bindingCacheKey(workspace, viewState);
	if (viewState.bindingVersion !== key) {
		viewState.bindingResults.clear();
		viewState.bindingLoads.clear();
		viewState.bindingVersion = key;
	}
	reconcileStreamSubscriptions(viewState, transport, workspace, tab, requestUpdate);
	for (const widget of tab.widgets) {
		const binding = primaryBinding(widget);
		if (!binding || viewState.bindingResults.has(widget.id) || viewState.bindingLoads.has(widget.id)) continue;
		if (binding.source === "stream") {
			const streamed = viewState.streamValues.get(widget.id);
			if (streamed) viewState.bindingResults.set(widget.id, streamed);
			continue;
		}
		viewState.bindingLoads.add(widget.id);
		(binding.source === "computed" ? resolveComputedForWidget(transport, widget, binding) : resolveBinding(transport, binding)).then((result) => {
			viewState.bindingResults.set(widget.id, result);
			viewState.bindingLoads.delete(widget.id);
			requestUpdate?.();
		});
	}
}
function gridMetrics(host) {
	return { width: (host instanceof HTMLElement ? host.querySelector(".dashboard-grid") : null)?.clientWidth ?? 0 };
}
/** Close the hidden-tabs overflow `<details>` on Escape. */
function onHiddenTabsKeydown(event) {
	if (event.key !== "Escape") return;
	const details = event.currentTarget.closest("details");
	if (details?.open) {
		event.preventDefault();
		details.open = false;
		details.querySelector("summary")?.focus();
	}
}
/** Arm a one-shot outside-click that closes the hidden-tabs overflow. */
function onHiddenTabsToggle(event) {
	const details = event.currentTarget;
	if (!details.open) return;
	const onOutside = (pointerEvent) => {
		if (pointerEvent.target instanceof Node && details.contains(pointerEvent.target)) return;
		details.open = false;
		document.removeEventListener("pointerdown", onOutside, true);
	};
	const onClosed = () => {
		if (!details.open) {
			document.removeEventListener("pointerdown", onOutside, true);
			details.removeEventListener("toggle", onClosed);
		}
	};
	document.addEventListener("pointerdown", onOutside, true);
	details.addEventListener("toggle", onClosed);
}
/**
* First-visit onboarding banner teaching how to add a tab. Dismissible +
* persisted — and only shown while the workspace is genuinely unfurnished
* (no widgets anywhere): a seeded/composed board doesn't need teaching.
*/
function renderOnboardingBanner(props, viewState, workspace, requestUpdate) {
	if (viewState.onboardingDismissed) return A;
	if (workspace.tabs.some((tab) => tab.widgets.length > 0)) return A;
	const dismiss = () => {
		viewState.onboardingDismissed = true;
		persistOnboardingDismissed(props.storage);
		requestUpdate();
	};
	return b`
    <div class="dashboard-onboarding" role="note" data-test-id="dashboard-onboarding">
      <span class="dashboard-onboarding__icon" aria-hidden="true">${icons.spark}</span>
      <div class="dashboard-onboarding__body">
        <div class="dashboard-onboarding__title">${t("dashboard.onboarding.title")}</div>
        <div class="dashboard-onboarding__sub">${t("dashboard.onboarding.primary")}</div>
        <div class="dashboard-onboarding__sub">
          ${t("dashboard.onboarding.secondary")}
          <code class="dashboard-onboarding__cmd">${t("dashboard.empty.onboardingCommand")}</code>
        </div>
      </div>
      <button
        class="dashboard-onboarding__dismiss"
        type="button"
        data-test-id="dashboard-onboarding-dismiss"
        aria-label=${t("common.dismiss")}
        @click=${dismiss}
      >
        ${icons.x}
      </button>
    </div>
  `;
}
function selectTab(props, state, workspace, slug) {
	state.activeSlug = resolveActiveSlug(workspace, slug);
	props.onNavigate?.(slug);
	props.onRequestUpdate?.();
}
/** Inline lock glyph for the private-tab marker (w4); icons.ts carries no lock. */
function lockGlyph() {
	return b`<svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
  </svg>`;
}
/**
* "Who's viewing this tab" indicator (w4): a live dot plus a count when more than
* one other operator is present. Rendered only when someone else is viewing — a
* solo operator sees no presence chrome.
*/
function renderTabPresence(viewers) {
	if (viewers <= 0) return A;
	const label = t("dashboard.tabs.presence", { count: String(viewers) });
	return b`
    <span
      class="dashboard-tab__presence"
      data-test-id="dashboard-tab-presence"
      title=${label}
      aria-label=${label}
    >
      <span class="dashboard-tab__presence-dot" aria-hidden="true"></span>
      ${viewers > 1 ? b`<span class="dashboard-tab__presence-count">${viewers}</span>` : A}
    </span>
  `;
}
/** One tab button in the strip. */
function renderTabButton(props, state, workspace, tab, active, viewers = 0) {
	return b`
    <button
      class="dashboard-tab ${active ? "dashboard-tab--active" : ""}"
      type="button"
      role="tab"
      aria-selected=${active ? "true" : "false"}
      data-test-id="dashboard-tab"
      data-ws=${tab.slug}
      @click=${() => selectTab(props, state, workspace, tab.slug)}
    >
      ${tab.icon && Object.hasOwn(icons, tab.icon) ? b`<span class="dashboard-tab__icon" aria-hidden="true"
              >${icons[tab.icon]}</span
            >` : A}
      <span class="dashboard-tab__label">${tab.title}</span>
      ${tab.visibility === "private" ? b`<span
              class="dashboard-tab__private"
              data-test-id="dashboard-tab-private"
              title=${t("dashboard.tabs.private")}
              aria-label=${t("dashboard.tabs.private")}
              >${lockGlyph()}</span
            >` : A}
      ${renderTabPresence(viewers)}
    </button>
  `;
}
/** Human label for an actor group header (w4). */
function tabGroupLabel(group) {
	if (group.kind === "agent") return t("dashboard.tabs.groupAgent", { agent: group.agentId ?? "agent" });
	return group.kind === "system" ? t("dashboard.tabs.groupSystem") : t("dashboard.tabs.groupUser");
}
/**
* Per-agent nesting (w4): render the visible tabs grouped by their `createdBy`
* provenance, each group foldable via a collapse toggle. When every visible tab
* shares one actor (the common case) the strip stays flat with no group chrome,
* preserving the single-workspace UX. Presence dots come from the host's presence
* store.
*/
function renderTabStrip(props, state, viewState, workspace) {
	const requestUpdate = () => props.onRequestUpdate?.();
	const tabs = visibleTabs(workspace);
	const groups = groupTabsByActor(tabs);
	const hidden = hiddenTabs(workspace);
	const grouped = groups.length > 1;
	const viewersOf = (slug) => presenceForTab(props.host, slug).length;
	return b`
    <nav class="dashboard-tabs" role="tablist" aria-label=${t("dashboard.tabs.label")}>
      ${grouped ? groups.map((group) => {
		const collapsed = viewState.collapsedTabGroups.has(group.key);
		const toggle = () => {
			if (collapsed) viewState.collapsedTabGroups.delete(group.key);
			else viewState.collapsedTabGroups.add(group.key);
			requestUpdate();
		};
		const label = tabGroupLabel(group);
		return b`
                <div
                  class="dashboard-tab-group ${collapsed ? "dashboard-tab-group--collapsed" : ""}"
                  data-test-id="dashboard-tab-group"
                  data-group=${group.key}
                >
                  <button
                    class="dashboard-tab-group__toggle"
                    type="button"
                    data-test-id="dashboard-tab-group-toggle"
                    aria-expanded=${collapsed ? "false" : "true"}
                    aria-label=${collapsed ? t("dashboard.tabs.expandGroup", { group: label }) : t("dashboard.tabs.collapseGroup", { group: label })}
                    @click=${toggle}
                  >
                    <span class="dashboard-tab-group__chevron" aria-hidden="true"
                      >${collapsed ? icons.chevronRight : icons.chevronDown}</span
                    >
                    <span class="dashboard-tab-group__label">${label}</span>
                    <span class="dashboard-tab-group__count">${group.tabs.length}</span>
                  </button>
                  ${collapsed ? A : group.tabs.map((tab) => renderTabButton(props, state, workspace, tab, tab.slug === state.activeSlug, viewersOf(tab.slug)))}
                </div>
              `;
	}) : tabs.map((tab) => renderTabButton(props, state, workspace, tab, tab.slug === state.activeSlug, viewersOf(tab.slug)))}
      ${hidden.length > 0 ? b`
              <details
                class="dashboard-tabs__hidden"
                @toggle=${onHiddenTabsToggle}
                @keydown=${onHiddenTabsKeydown}
              >
                <summary class="dashboard-tab dashboard-tab--overflow">
                  <span class="dashboard-tab__icon" aria-hidden="true">${icons.eyeOff}</span>
                  <span class="dashboard-tab__label"
                    >${t("dashboard.tabs.hidden", { count: String(hidden.length) })}</span
                  >
                </summary>
                <div class="dashboard-tabs__hidden-menu" role="menu">
                  ${hidden.map((tab) => b`
                      <button
                        class="dashboard-tabs__hidden-item"
                        type="button"
                        role="menuitem"
                        @click=${() => selectTab(props, state, workspace, tab.slug)}
                      >
                        ${tab.title}
                      </button>
                    `)}
                </div>
              </details>
            ` : A}
    </nav>
  `;
}
/** Load `widget.json` manifests for the APPROVED custom widgets on the active tab. */
function ensureManifests(viewState, props, workspace, tab) {
	const basePath = props.basePath ?? "";
	for (const widget of tab.widgets) {
		const name = customWidgetName(widget.kind);
		if (!name || customWidgetStatus(workspace, widget.kind) !== "approved" || viewState.manifestCache.has(name) || viewState.manifestLoads.has(name)) continue;
		viewState.manifestLoads.add(name);
		loadWidgetManifestView(basePath, name).then((manifest) => {
			viewState.manifestLoads.delete(name);
			if (manifest) {
				viewState.manifestCache.set(name, manifest);
				props.onRequestUpdate?.();
			}
		});
	}
}
/**
* Wire the action-form builtin's prompt dispatch to the shared confirm + rate-limit
* gate — the SAME `dispatchRateLimitedPrompt` the custom-widget bridge uses, with the
* same `confirm` fallback and the same `chat.send` path. No new privilege.
*/
function makeBuiltinDispatchPrompt(props) {
	const transport = props.transport;
	const sessionKey = props.sessionKey ?? "main";
	return ({ widgetKey, text }) => dispatchRateLimitedPrompt({
		widgetKey,
		text,
		confirmPrompt: async (prompt) => {
			if (props.confirm) return await props.confirm(prompt);
			return typeof window !== "undefined" ? window.confirm(prompt) : false;
		},
		sendPrompt: async (prompt) => {
			if (!transport) throw new Error("Not connected.");
			await transport.request("chat.send", {
				sessionKey,
				message: prompt,
				deliver: false
			});
		}
	});
}
/**
* Builds the builtin-widget context for ONE widget. The write-back `state` accessor
* is bound to THIS widget's own `widget.id` (host-tracked, never child-supplied), so
* a stateful builtin (notes) can only read/write its own state; it is present only
* when a transport exists. `dispatchPrompt` (action-form) and `approvals` are the
* shared, workspace-scoped seams.
*/
function buildBuiltinContext(props, state, workspace, widget) {
	const transport = props.transport;
	const context = {
		embed: embedContext(props.embed),
		dispatchPrompt: makeBuiltinDispatchPrompt(props),
		onActionError: (message) => {
			state.actionError = message;
			props.onRequestUpdate?.();
		},
		approvals: buildApprovalsSource(workspace, (name, decision) => void approveWidget(state, transport, {
			name,
			decision
		}), (name, decision, options) => void approveCapability(state, transport, {
			name,
			decision,
			...options?.tools !== void 0 ? { tools: options.tools } : {},
			...options?.autoConfirm !== void 0 ? { autoConfirm: options.autoConfirm } : {},
			...options?.expiresAt !== void 0 ? { expiresAt: options.expiresAt } : {}
		})),
		registryPending: pendingWidgetNames(workspace)
	};
	if (transport) {
		context.state = createBuiltinStateAccessor(transport, widget.id);
		context.chat = makeBuiltinChatSeam(transport, props.sessionKey ?? "main");
		context.approveWidget = (name, decision) => void approveWidget(state, transport, {
			name,
			decision
		});
		context.actions = makeBuiltinActionsSeam(transport, props.operator === true);
	}
	return context;
}
/**
* Build the external-tool action seam (SPEC §17 v2 / §18) for the action-button and
* tool-mode action-form widgets. `invoke` maps the engine response to the outcome
* union (a parked mutation → `pending`, a readOnly execution → `result`); `confirm`/
* `deny` are attached ONLY for the local operator, so a networked client's widget
* renders the confirm affordance disabled-with-reason (the server enforces the same —
* invariant #5). `subscribe` multiplexes the `dashboard.action.changed` broadcast over
* the transport's existing event stream (no new socket).
*/
function makeBuiltinActionsSeam(transport, operator) {
	const seam = {
		invoke: async (params) => {
			const response = await transport.request("dashboard.action.invoke", params);
			if (isRecord$4(response) && response.pending === true) return {
				kind: "pending",
				id: typeof response.id === "string" ? response.id : "",
				expiresAt: typeof response.expiresAt === "string" ? response.expiresAt : ""
			};
			return {
				kind: "result",
				result: response
			};
		},
		subscribe: (listener) => transport.addEventListener("dashboard.action.changed", (payload) => {
			if (isRecord$4(payload) && typeof payload.id === "string") listener({
				id: payload.id,
				status: payload.status,
				connector: typeof payload.connector === "string" ? payload.connector : "",
				tool: typeof payload.tool === "string" ? payload.tool : ""
			});
		})
	};
	if (operator) {
		seam.confirm = async (id) => {
			const response = await transport.request("dashboard.action.confirm", { id });
			return { result: isRecord$4(response) && "result" in response ? response.result : response };
		};
		seam.deny = async (id) => {
			await transport.request("dashboard.action.deny", { id });
		};
	}
	return seam;
}
/** Names of `custom:` widgets currently `pending` approval (chat inline approval card). */
function pendingWidgetNames(workspace) {
	return Object.entries(workspace.widgetsRegistry).filter(([, entry]) => entry.status === "pending").map(([name]) => name);
}
/**
* Build the `builtin:chat` control-plane seam (SPEC §14): all four `chat.*` methods
* bound to a single `sessionKey`, with the broadcast bus (`CHAT_EVENT`) and the
* history ring filtered to that key. The renderer knows nothing about providers —
* this seam is the whole coupling to the control plane.
*/
function makeBuiltinChatSeam(transport, sessionKey) {
	const belongsToSession = (event) => event.sessionKey === sessionKey;
	return {
		send: async (message) => {
			return { turnId: (await transport.request("chat.send", {
				sessionKey,
				message
			})).turnId };
		},
		abort: async (turnId) => {
			await transport.request("chat.abort", {
				sessionKey,
				turnId
			});
		},
		history: async () => {
			return ((await transport.request("chat.history.get", { sessionKey })).events ?? []).filter(belongsToSession);
		},
		subscribe: (listener) => transport.addEventListener(CHAT_EVENT, (payload) => {
			const event = payload;
			if (event && belongsToSession(event)) listener(event);
		})
	};
}
/** Widget-id-bound write-back accessor over the transport (notes builtin). */
function createBuiltinStateAccessor(transport, widgetId) {
	return {
		get: () => transport.request("dashboard.widget.state.get", { widgetId }),
		set: (blob) => transport.request("dashboard.widget.state.set", {
			widgetId,
			state: blob
		})
	};
}
/** Builds the custom-widget context for one `custom:<name>` widget, or null. */
function buildCustomContext(props, state, viewState, workspace, widget, tabSlug) {
	const name = customWidgetName(widget.kind);
	if (!name) return null;
	return {
		status: customWidgetStatus(workspace, widget.kind),
		manifest: viewState.manifestCache.get(name) ?? null,
		host: {
			transport: props.transport,
			basePath: props.basePath ?? "",
			sessionKey: props.sessionKey ?? "main",
			tabSlug,
			...props.confirm ? { confirmPrompt: props.confirm } : {}
		},
		onApprove: () => void approveWidget(state, props.transport, {
			name,
			decision: "approved"
		}),
		onReject: () => void approveWidget(state, props.transport, {
			name,
			decision: "rejected"
		})
	};
}
/** Loaded snapshot bodies as an ordered list, for the blame first-seen lookup (m2). */
function loadedHistorySnapshots(viewState) {
	return [...viewState.history.snapshots.entries()].map(([version, workspace]) => ({
		version,
		workspace
	}));
}
/**
* Build the blame line for a widget (m2), or undefined when it carries no
* provenance. The first-seen version is recovered from whatever history snapshots
* are already loaded (the panel loads them); the logbook link is offered only for
* agent authors when a link is derivable.
*/
function computeWidgetBlame(props, viewState, widget) {
	const actor = widget.createdBy;
	if (!actor) return;
	const agentId = dashboardAgentProvenance(actor);
	const seen = firstSeenVersion(widget.id, loadedHistorySnapshots(viewState));
	return {
		actor,
		agentId,
		...seen !== void 0 ? { firstSeenVersion: seen } : {},
		...agentId ? { logbookHref: props.logbookHref ?? null } : {}
	};
}
/** Fetch (or refetch) the ring metadata and auto-select the newest snapshot (m2). */
async function refreshHistoryList(props, viewState) {
	const requestUpdate = () => props.onRequestUpdate?.();
	const history = viewState.history;
	history.loading = true;
	history.error = null;
	requestUpdate();
	try {
		const entries = await loadHistoryList(props.transport);
		history.entries = entries;
		if (entries.length > 0 && history.selectedVersion === null) history.selectedVersion = entries[0].version;
		history.error = null;
	} catch (err) {
		history.error = err instanceof Error ? err.message : String(err);
	} finally {
		history.loading = false;
		requestUpdate();
	}
	if (history.selectedVersion !== null) await ensureHistorySnapshot(props, viewState, history.selectedVersion);
}
/** Lazily load one snapshot body into the per-host cache (m2). */
async function ensureHistorySnapshot(props, viewState, version) {
	const history = viewState.history;
	if (history.snapshots.has(version)) return;
	try {
		const workspace = await loadHistorySnapshot(props.transport, version);
		if (workspace) {
			history.snapshots.set(version, workspace);
			props.onRequestUpdate?.();
		}
	} catch (err) {
		history.error = err instanceof Error ? err.message : String(err);
		props.onRequestUpdate?.();
	}
}
/** Open the time-travel panel and load the ring (m2). */
function openHistory(props, viewState) {
	viewState.history.open = true;
	viewState.history.confirmRestore = false;
	refreshHistoryList(props, viewState);
	props.onRequestUpdate?.();
}
/** Close the time-travel panel; loaded snapshots stay cached for the blame line (m2). */
function closeHistory(props, viewState) {
	viewState.history.open = false;
	viewState.history.confirmRestore = false;
	props.onRequestUpdate?.();
}
/** Select a history version, loading its body on demand (m2). */
function selectHistoryVersion(props, viewState, version) {
	viewState.history.selectedVersion = version;
	ensureHistorySnapshot(props, viewState, version);
	props.onRequestUpdate?.();
}
/** Grid rect for a freshly-installed widget: a default cell below existing rows (w3). */
function installPlacementGrid(tab, bundle) {
	const manifest = bundle.manifest;
	const preferred = manifest.preferredSize && typeof manifest.preferredSize === "object" ? manifest.preferredSize : {};
	const w = Math.min(12, Math.max(1, Number(preferred.w) || 6));
	const h = Math.max(1, Number(preferred.h) || 4);
	return {
		x: 0,
		y: (tab?.widgets ?? []).reduce((max, widget) => {
			const bottom = widget.grid.y + widget.grid.h;
			return bottom > max ? bottom : max;
		}, 0),
		w,
		h
	};
}
function renderGrid(props, state, viewState, workspace, tab) {
	ensureBindings(viewState, props.transport, workspace, tab, props.onRequestUpdate ?? null);
	ensureManifests(viewState, props, workspace, tab);
	if (tab.widgets.length === 0) return b`
      <div class="dashboard-empty dashboard-empty--tab" data-test-id="dashboard-empty-tab">
        <span class="dashboard-empty__icon" aria-hidden="true">${icons.plus}</span>
        <div class="dashboard-empty__title">${t("dashboard.empty.tabTitle")}</div>
        <div class="dashboard-empty__sub">${t("dashboard.empty.tabSubtitle")}</div>
      </div>
    `;
	if (tab.layout === "full") return renderFullBleed(props, state, viewState, workspace, tab);
	const callbacks = makeCallbacks(props, state, viewState, tab);
	const rows = gridRowCount(tab.widgets);
	const minHeight = rows * 56 + Math.max(0, rows - 1) * 12;
	const multiAgent = distinctAgentActors(workspace).length >= 2;
	return b`
    <div class="dashboard-grid" style="min-height: ${minHeight}px" data-test-id="dashboard-grid">
      ${tab.widgets.map((widget) => {
		const custom = buildCustomContext(props, state, viewState, workspace, widget, tab.slug);
		const blame = computeWidgetBlame(props, viewState, widget);
		const drag = viewState.drag;
		const dragging = drag?.widgetId === widget.id;
		const dragTransform = dragging && drag.mode === "move" ? `translate(${drag.pointerDx}px, ${drag.pointerDy}px)` : void 0;
		const agentChip = multiAgent && widget.createdBy ? agentChipFor(widget.createdBy, viewState.highlightedAgent) : null;
		return renderWidgetCell({
			widget,
			binding: viewState.bindingResults.get(widget.id) ?? null,
			...blame ? { blame } : {},
			menuOpen: viewState.openMenuWidgetId === widget.id,
			pending: state.pendingWidgetIds.has(widget.id),
			dragging,
			...dragTransform ? { dragTransform } : {},
			builtinContext: buildBuiltinContext(props, state, workspace, widget),
			callbacks,
			...custom ? { custom } : {},
			...agentChip ? { agentChip } : {}
		});
	})}
      ${renderDragGhost(viewState, tab)}
    </div>
  `;
}
/**
* Full-bleed layout (w3): render the tab's FIRST widget filling the whole content
* area with no grid chrome (no placement styles, no drag/resize handles). The
* widget body reuses the same builtin/custom render path (and per-cell error
* boundary) as the grid, so bindings, the sandboxed iframe host, and the approval
* gate all behave identically — only the surrounding layout differs.
*/
function renderFullBleed(props, state, viewState, workspace, tab) {
	const widget = tab.widgets[0];
	const callbacks = makeCallbacks(props, state, viewState, tab);
	const custom = buildCustomContext(props, state, viewState, workspace, widget, tab.slug);
	return b`
    <div class="dashboard-fullbleed" data-test-id="dashboard-fullbleed" data-widget-id=${widget.id}>
      ${renderWidgetBody(widget, viewState.bindingResults.get(widget.id) ?? null, buildBuiltinContext(props, state, workspace, widget), callbacks, custom ?? void 0)}
    </div>
  `;
}
/** Snapped drop-target ghost for the active move/resize drag. */
function renderDragGhost(viewState, tab) {
	const drag = viewState.drag;
	if (!drag) return A;
	return b`
    <div
      class="dashboard-ghost ${collides(drag.ghostRect, tab.widgets, drag.widgetId) ? "dashboard-ghost--invalid" : ""}"
      style=${gridPlacementStyle(drag.ghostRect)}
      aria-hidden="true"
      data-test-id="dashboard-drag-ghost"
    ></div>
  `;
}
function makeCallbacks(props, state, viewState, tab) {
	const requestUpdate = () => props.onRequestUpdate?.();
	const commitDrag = (widget, event, mode) => {
		const metrics = gridMetrics(props.host);
		if (metrics.width <= 0) return;
		const drag = beginDrag({
			widget,
			mode,
			clientX: event.clientX,
			clientY: event.clientY,
			metrics
		});
		viewState.drag = drag;
		const target = event.target;
		try {
			target.setPointerCapture?.(event.pointerId);
		} catch {}
		let settled = false;
		const teardown = () => {
			window.removeEventListener("pointermove", onMove);
			window.removeEventListener("pointerup", onUp);
		};
		const cancel = () => {
			if (settled) return;
			settled = true;
			teardown();
			viewState.drag = null;
			requestUpdate();
		};
		const onMove = (moveEvent) => {
			updateDrag(drag, moveEvent.clientX, moveEvent.clientY);
			requestUpdate();
		};
		const onUp = () => {
			if (settled) return;
			settled = true;
			teardown();
			clearActiveDrag(props.host);
			const resolved = resolveDrop({
				requested: drag.ghostRect,
				widgets: tab.widgets,
				widgetId: widget.id
			});
			viewState.drag = null;
			requestUpdate();
			if (resolved && (resolved.x !== widget.grid.x || resolved.y !== widget.grid.y || resolved.w !== widget.grid.w || resolved.h !== widget.grid.h)) moveWidget(state, props.transport, {
				slug: tab.slug,
				widgetId: widget.id,
				grid: resolved
			});
		};
		window.addEventListener("pointermove", onMove);
		window.addEventListener("pointerup", onUp);
		registerActiveDrag(props.host, cancel);
	};
	return {
		onToggleCollapse: (widget) => void setWidgetCollapsed(state, props.transport, {
			slug: tab.slug,
			widgetId: widget.id,
			collapsed: !widget.collapsed
		}),
		onToggleMenu: (widget) => {
			viewState.openMenuWidgetId = viewState.openMenuWidgetId === widget.id ? null : widget.id;
			requestUpdate();
		},
		onHide: (widget) => {
			viewState.openMenuWidgetId = null;
			hideWidget(state, props.transport, {
				slug: tab.slug,
				widgetId: widget.id
			});
		},
		onRemove: (widget) => {
			viewState.openMenuWidgetId = null;
			removeWidgetFromTab(state, props.transport, {
				slug: tab.slug,
				widgetId: widget.id
			});
		},
		onEditTitle: (widget) => {
			viewState.openMenuWidgetId = null;
			viewState.dialog = {
				kind: "editTitle",
				slug: tab.slug,
				widgetId: widget.id,
				title: widget.title
			};
			requestUpdate();
		},
		onMoveToTab: (widget) => {
			viewState.openMenuWidgetId = null;
			viewState.dialog = {
				kind: "moveToTab",
				slug: tab.slug,
				widgetId: widget.id
			};
			requestUpdate();
		},
		onPin: (widget) => {
			viewState.openMenuWidgetId = null;
			pinWidget(state, props.transport, {
				slug: tab.slug,
				widgetId: widget.id
			});
		},
		onMovePointerDown: (widget, event) => {
			if (event.button !== 0) return;
			event.preventDefault();
			commitDrag(widget, event, "move");
		},
		onResizePointerDown: (widget, event) => {
			if (event.button !== 0) return;
			event.preventDefault();
			event.stopPropagation();
			commitDrag(widget, event, "resize");
		},
		onKeyboardNudge: (widget, mode, direction) => {
			const resolved = resolveDrop({
				requested: nudgeRect(widget.grid, mode, direction),
				widgets: tab.widgets,
				widgetId: widget.id
			});
			if (resolved) moveWidget(state, props.transport, {
				slug: tab.slug,
				widgetId: widget.id,
				grid: resolved
			});
		}
	};
}
/** Minimal themed modal (Escape/backdrop cancel) replacing window.prompt() flows. */
function renderModal(label, onCancel, body) {
	const onBackdrop = (event) => {
		if (event.target === event.currentTarget) onCancel();
	};
	const onKeydown = (event) => {
		if (event.key === "Escape") {
			event.preventDefault();
			onCancel();
		}
	};
	return b`
    <div
      class="bs-modal"
      role="dialog"
      aria-modal="true"
      aria-label=${label}
      data-test-id="bs-modal"
      @click=${onBackdrop}
      @keydown=${onKeydown}
    >
      <div class="bs-modal__card">${body}</div>
    </div>
  `;
}
/** Themed edit-title / move-to-tab dialog, replacing window.prompt(). */
function renderDialog(props, state, viewState) {
	const dialog = viewState.dialog;
	if (!dialog) return A;
	const requestUpdate = () => props.onRequestUpdate?.();
	const close = () => {
		viewState.dialog = null;
		requestUpdate();
	};
	if (dialog.kind === "editTitle") {
		const title = t("dashboard.widget.editTitleTitle");
		const submit = (event) => {
			event.preventDefault();
			const next = event.currentTarget.querySelector("input[name='dashboard-widget-title']")?.value.trim() ?? "";
			if (next && next !== dialog.title) updateWidgetTitle(state, props.transport, {
				slug: dialog.slug,
				widgetId: dialog.widgetId,
				title: next
			});
			close();
		};
		return renderModal(title, close, b`
        <form class="bs-dialog" @submit=${submit}>
          <div class="bs-dialog__title">${title}</div>
          <input
            class="bs-dialog__input"
            type="text"
            name="dashboard-widget-title"
            data-test-id="dashboard-edit-title-input"
            .value=${dialog.title}
            aria-label=${t("dashboard.widget.editTitleLabel")}
          />
          <div class="bs-dialog__actions">
            <button class="bs-btn bs-btn--primary" type="submit">${t("common.save")}</button>
            <button class="bs-btn" type="button" @click=${close}>${t("common.cancel")}</button>
          </div>
        </form>
      `);
	}
	const title = t("dashboard.widget.moveToTabTitle");
	const targets = state.workspace ? orderedTabs(state.workspace).filter((candidate) => candidate.slug !== dialog.slug) : [];
	const submit = (event) => {
		event.preventDefault();
		const toSlug = event.currentTarget.querySelector("select[name='dashboard-move-target']")?.value ?? "";
		if (toSlug && toSlug !== dialog.slug) moveWidgetToTab(state, props.transport, {
			fromSlug: dialog.slug,
			toSlug,
			widgetId: dialog.widgetId
		});
		close();
	};
	return renderModal(title, close, b`
      <form class="bs-dialog" @submit=${submit}>
        <div class="bs-dialog__title">${title}</div>
        ${targets.length === 0 ? b`<div class="bs-dialog__sub">${t("dashboard.widget.moveToTabEmpty")}</div>` : b`<select
                class="bs-dialog__input"
                name="dashboard-move-target"
                data-test-id="dashboard-move-target"
                aria-label=${title}
              >
                ${targets.map((candidate) => b`<option value=${candidate.slug}>${candidate.title}</option>`)}
              </select>`}
        <div class="bs-dialog__actions">
          <button class="bs-btn bs-btn--primary" type="submit" ?disabled=${targets.length === 0}>
            ${t("dashboard.widget.menu.moveToTab")}
          </button>
          <button class="bs-btn" type="button" @click=${close}>${t("common.cancel")}</button>
        </div>
      </form>
    `);
}
/**
* Render the reference view for `props`. The element owns lifecycle (load /
* subscribe / poll) via the host store, keyed on `props.host`.
*/
function renderBoardstateView(props) {
	setBoardstateStrings(props.strings);
	const state = getDashboardState(props.host);
	const viewState = getViewState(props.host, props.storage);
	state.requestUpdate = props.onRequestUpdate ?? null;
	syncMenuDismiss(props.host, viewState, () => props.onRequestUpdate?.());
	const active = props.connected;
	subscribeToDashboardEvents(props.host, state, active ? props.transport : null);
	startBindingPolling(props.host, active ? props.transport : null, () => {
		bumpBoardstateDataVersion(props.host);
		if (active && state.activeSlug) pingPresence(props.host, props.transport, state.activeSlug);
		props.onRequestUpdate?.();
	});
	if (active && !state.loaded && !state.loading && !state.error) loadWorkspace(state, props.transport, { requestedSlug: props.initialTab ?? null });
	if (active && state.activeSlug && viewState.lastPresenceSlug !== state.activeSlug) {
		viewState.lastPresenceSlug = state.activeSlug;
		pingPresence(props.host, props.transport, state.activeSlug);
	}
	return b`
    <section class="dashboard" data-test-id="dashboard">
      ${state.actionError ? b`<div class="callout danger dashboard__toast" role="alert">
              ${state.actionError}
            </div>` : A}
      ${renderBody(props, state, viewState)} ${renderDialog(props, state, viewState)}
      ${renderHistoryPanel(props, state, viewState)} ${renderGalleryDialog(props, state, viewState)}
    </section>
  `;
}
function renderBody(props, state, viewState) {
	if (state.error) return b`
      <div class="card lazy-view-state" role="alert">
        <div class="card-title">${t("dashboard.error.title")}</div>
        <div class="card-sub">${t("dashboard.error.subtitle")}</div>
        <details class="dashboard-error-detail">
          <summary>${t("dashboard.error.detailSummary")}</summary>
          <div class="dashboard-error-detail__text">${state.error}</div>
        </details>
        <button
          class="bs-btn bs-btn--small"
          type="button"
          @click=${() => void loadWorkspace(state, props.transport)}
        >
          ${t("common.reload")}
        </button>
      </div>
    `;
	const workspace = state.workspace;
	if (!workspace) return b`
      <div class="dashboard-skeleton" role="status" aria-label=${t("common.loading")}>
        ${[
		0,
		1,
		2,
		3,
		4,
		5
	].map(() => b`<div class="dashboard-skeleton__card"></div>`)}
      </div>
    `;
	if (workspace.tabs.length === 0) return b`
      <div class="dashboard-empty dashboard-empty--onboarding" data-test-id="dashboard-empty">
        <div class="dashboard-empty__title">${t("dashboard.empty.onboardingTitle")}</div>
        <div class="dashboard-empty__sub">${t("dashboard.empty.onboardingSubtitle")}</div>
        <code class="dashboard-empty__cmd">${t("dashboard.empty.onboardingCommand")}</code>
      </div>
    `;
	const tab = findTab(workspace, state.activeSlug) ?? visibleTabs(workspace)[0];
	if (!tab) return b`<div class="card lazy-view-state" role="status">
      <div class="card-sub">${t("dashboard.empty.noVisibleTabs")}</div>
    </div>`;
	return b`
    ${renderWorkspacesHeader(props, state, viewState, tab)}
    ${renderOnboardingBanner(props, viewState, workspace, () => props.onRequestUpdate?.())}
    ${renderTabStrip(props, state, viewState, workspace)}
    ${renderAgentFilterBar(props, viewState, workspace)}
    ${renderGrid(props, state, viewState, workspace, tab)}
  `;
}
/**
* The per-agent provenance filter (SPEC §17.3, #59): a row of deterministically-coloured
* agent chips shown ONLY on a multi-agent board (≥2 distinct agent authors). Clicking a
* chip highlights that agent's widgets (dimming the rest); clicking the active chip (or
* "All") clears the filter. A single-agent / operator board renders nothing here.
*/
function renderAgentFilterBar(props, viewState, workspace) {
	const actors = distinctAgentActors(workspace);
	if (actors.length < 2) {
		viewState.highlightedAgent = null;
		return A;
	}
	const setHighlight = (actor) => {
		viewState.highlightedAgent = viewState.highlightedAgent === actor ? null : actor;
		props.onRequestUpdate?.();
	};
	const active = viewState.highlightedAgent;
	return b`
    <div
      class="dashboard-agent-filter"
      data-test-id="dashboard-agent-filter"
      role="group"
      aria-label=${t("dashboard.agentFilter.label")}
    >
      <span class="dashboard-agent-filter__label">${t("dashboard.agentFilter.label")}</span>
      <button
        class="dashboard-agent-filter__chip ${active === null ? "dashboard-agent-filter__chip--active" : ""}"
        type="button"
        data-test-id="dashboard-agent-filter-all"
        aria-pressed=${active === null ? "true" : "false"}
        @click=${() => setHighlight(null)}
      >
        ${t("dashboard.agentFilter.all")}
      </button>
      ${actors.map((actor) => {
		const agentId = dashboardAgentProvenance(actor) ?? actor;
		const isActive = active === actor;
		return b`<button
          class="dashboard-agent-filter__chip dashboard-agent-filter__chip--agent ${isActive ? "dashboard-agent-filter__chip--active" : ""}"
          type="button"
          style="--dashboard-agent-hue: ${agentHue(actor)}"
          data-agent=${actor}
          data-test-id="dashboard-agent-filter-chip"
          aria-pressed=${isActive ? "true" : "false"}
          title=${t("dashboard.widget.agentChipTooltip", { agent: actor })}
          @click=${() => setHighlight(actor)}
        >
          ${shortAgentId(agentId)}
        </button>`;
	})}
    </div>
  `;
}
/** Trigger a browser download of `json` under `filename` (no-op outside a document). */
function downloadWorkspaceJson(filename, json) {
	if (typeof document === "undefined" || typeof URL.createObjectURL !== "function") return;
	const blob = new Blob([json], { type: "application/json" });
	const url = URL.createObjectURL(blob);
	const anchor = document.createElement("a");
	anchor.href = url;
	anchor.download = filename;
	document.body.append(anchor);
	anchor.click();
	anchor.remove();
	URL.revokeObjectURL(url);
}
/** Export the full workspace to a downloadable JSON file; a failure surfaces a toast (w5). */
function runWorkspaceExport(props, state) {
	exportWorkspace(props.transport).then((file) => downloadWorkspaceJson(file.filename, file.json)).catch((err) => {
		state.actionError = err instanceof Error ? err.message : String(err);
		props.onRequestUpdate?.();
	});
}
/** Read the chosen file and apply it via importWorkspace (custom widgets → pending) (w5). */
function onWorkspaceImportChange(props, state, event) {
	const input = event.currentTarget;
	const file = input.files?.[0];
	input.value = "";
	if (!file) return;
	file.text().then((text) => importWorkspace(state, props.transport, text));
}
/** Open the widget gallery dialog seeded with the remembered registry URL (w3). */
function openGallery(props, viewState) {
	viewState.gallery = {
		indexUrl: readGalleryUrl(props.storage),
		mode: "widgets",
		entries: null,
		selected: null,
		recipes: null,
		selectedRecipe: null,
		busy: false,
		error: null
	};
	props.onRequestUpdate?.();
}
/**
* Page-header treatment for the active workspace tab. Carries the tab-level actions:
* the widget-gallery opener + full-bleed toggle (w3), the time-travel toggle (m2),
* and the export/import distribution controls (w5).
*/
function renderWorkspacesHeader(props, state, viewState, tab) {
	const isFull = tab.layout === "full";
	const toggleLayout = () => void setTabLayout(state, props.transport, {
		slug: tab.slug,
		layout: isFull ? "grid" : "full"
	});
	return b`
    <div class="dashboard-page-header" data-test-id="dashboard-page-header">
      <div class="dashboard-page-header__titles">
        <div class="page-title">${tab.title}</div>
        <div class="page-sub">${t("dashboard.header.subtitle")}</div>
      </div>
      <div
        class="dashboard-page-header__actions dashboard-toolbar"
        data-test-id="dashboard-toolbar"
      >
        <button
          class="bs-btn bs-btn--small"
          type="button"
          data-test-id="dashboard-gallery-open"
          title=${t("dashboard.gallery.open")}
          @click=${() => openGallery(props, viewState)}
        >
          <span class="dashboard-page-header__action-icon" aria-hidden="true">${icons.puzzle}</span>
          ${t("dashboard.gallery.open")}
        </button>
        <button
          class="bs-btn bs-btn--small ${isFull ? "bs-btn--primary" : ""}"
          type="button"
          data-test-id="dashboard-fullbleed-toggle"
          aria-pressed=${isFull ? "true" : "false"}
          title=${isFull ? t("dashboard.header.fullBleedExit") : t("dashboard.header.fullBleedEnter")}
          @click=${toggleLayout}
        >
          <span class="dashboard-page-header__action-icon" aria-hidden="true"
            >${isFull ? icons.minimize : icons.maximize}</span
          >
          ${isFull ? t("dashboard.header.fullBleedExit") : t("dashboard.header.fullBleedEnter")}
        </button>
        <button
          class="bs-btn bs-btn--small dashboard-history__toggle"
          type="button"
          data-test-id="dashboard-history-toggle"
          @click=${() => openHistory(props, viewState)}
        >
          ${icons.clock} ${t("dashboard.history.open")}
        </button>
        <button
          class="bs-btn bs-btn--small"
          type="button"
          data-test-id="dashboard-export"
          title=${t("dashboard.distribution.exportTitle")}
          @click=${() => runWorkspaceExport(props, state)}
        >
          ${t("dashboard.distribution.export")}
        </button>
        <button
          class="bs-btn bs-btn--small"
          type="button"
          data-test-id="dashboard-import"
          title=${t("dashboard.distribution.importTitle")}
          @click=${(event) => event.currentTarget.parentElement?.querySelector("input[type=\"file\"]")?.click()}
        >
          ${t("dashboard.distribution.import")}
        </button>
        <input
          type="file"
          accept="application/json,.json"
          hidden
          data-test-id="dashboard-import-input"
          @change=${(event) => onWorkspaceImportChange(props, state, event)}
        />
      </div>
    </div>
  `;
}
/** Compact relative time for a history entry's ISO timestamp (m2, view-local). */
function formatRelativeTimestamp(iso) {
	const ms = Date.parse(iso);
	if (!Number.isFinite(ms)) return iso;
	const seconds = Math.round((Date.now() - ms) / 1e3);
	if (seconds < 60) return "just now";
	const minutes = Math.round(seconds / 60);
	if (minutes < 60) return `${minutes}m ago`;
	const hours = Math.round(minutes / 60);
	if (hours < 24) return `${hours}h ago`;
	const days = Math.round(hours / 24);
	if (days < 7) return `${days}d ago`;
	try {
		return new Intl.DateTimeFormat("en-US", {
			month: "short",
			day: "numeric"
		}).format(new Date(ms));
	} catch {
		return iso;
	}
}
/**
* Read-only time-travel panel (m2): the version list on the left, a selected
* snapshot's preview + diff-vs-current + restore on the right. Restore reuses the
* existing single-step undo, so it is offered only for the newest snapshot.
*/
function renderHistoryPanel(props, state, viewState) {
	const history = viewState.history;
	if (!history.open) return A;
	const title = t("dashboard.history.title");
	const selected = history.selectedVersion !== null ? history.snapshots.get(history.selectedVersion) : void 0;
	const newestVersion = history.entries[0]?.version ?? null;
	return renderModal(title, () => closeHistory(props, viewState), b`
      <div class="dashboard-history" data-test-id="dashboard-history">
        <div class="dashboard-history__header">
          <div class="card-title">${title}</div>
          <div class="card-sub">${t("dashboard.history.subtitle")}</div>
        </div>
        ${history.error ? b`<div class="callout danger" role="alert">${history.error}</div>` : A}
        <div class="dashboard-history__body">
          ${renderHistoryList(props, viewState, newestVersion)}
          <div class="dashboard-history__detail">
            ${history.selectedVersion === null ? b`<div class="card-sub">${t("dashboard.history.emptyDetail")}</div>` : renderHistoryDetail(props, state, viewState, history.selectedVersion, selected)}
          </div>
        </div>
        <div class="bs-dialog__actions">
          <button class="bs-btn" type="button" @click=${() => closeHistory(props, viewState)}>
            ${t("common.close")}
          </button>
        </div>
      </div>
    `);
}
/**
* Compose the compact per-row change label from a store-computed summary, e.g.
* "+2 · 1 moved". A summary with no tracked layout change (a props-only or
* collapse mutation, which the diff doesn't count) still renders "Other edit" so
* the row reads as a real, distinct change rather than an empty repeat.
*/
function historySummaryLabel(summary) {
	const parts = [];
	if (summary.added > 0) parts.push(t("dashboard.history.summary.added", { count: String(summary.added) }));
	if (summary.removed > 0) parts.push(t("dashboard.history.summary.removed", { count: String(summary.removed) }));
	if (summary.moved > 0) parts.push(t("dashboard.history.summary.moved", { count: String(summary.moved) }));
	if (summary.retitled > 0) parts.push(t("dashboard.history.summary.retitled", { count: String(summary.retitled) }));
	if (summary.tabsChanged > 0) parts.push(t("dashboard.history.summary.tabs", { count: String(summary.tabsChanged) }));
	return parts.length > 0 ? parts.join(" · ") : t("dashboard.history.summary.minor");
}
/** The change-summary line under a history row's version (m2 polish, #4). */
function renderHistoryChangeSummary(summary) {
	if (!summary) return A;
	return b`<span class="dashboard-history__change">
    <span class="dashboard-history__change-label">${historySummaryLabel(summary)}</span>
  </span>`;
}
function renderHistoryList(props, viewState, newestVersion) {
	const history = viewState.history;
	if (history.loading && history.entries.length === 0) return b`<div class="dashboard-history__list">
      <div class="card-sub">${t("common.loading")}</div>
    </div>`;
	if (history.entries.length === 0) return b`<div class="dashboard-history__list">
      <div class="card-sub">${t("dashboard.history.empty")}</div>
    </div>`;
	return b`
    <ul class="dashboard-history__list" role="listbox" aria-label=${t("dashboard.history.title")}>
      ${history.entries.map((entry) => {
		const active = entry.version === history.selectedVersion;
		return b`
          <li>
            <button
              class="dashboard-history__item ${active ? "dashboard-history__item--active" : ""}"
              type="button"
              role="option"
              aria-selected=${active ? "true" : "false"}
              data-test-id="dashboard-history-item"
              @click=${() => selectHistoryVersion(props, viewState, entry.version)}
            >
              <span class="dashboard-history__version"
                >${t("dashboard.history.version", { version: String(entry.version) })}</span
              >
              ${renderHistoryChangeSummary(entry.summary)}
              <span class="dashboard-history__time">${formatRelativeTimestamp(entry.savedAt)}</span>
              ${entry.version === newestVersion ? b`<span class="dashboard-history__latest"
                      >${t("dashboard.history.latest")}</span
                    >` : A}
            </button>
          </li>
        `;
	})}
    </ul>
  `;
}
function renderHistoryDetail(props, state, viewState, version, snapshot) {
	const history = viewState.history;
	const current = state.workspace;
	const isNewest = version === (history.entries[0]?.version ?? null);
	if (!snapshot) return b`<div class="card-sub" data-test-id="dashboard-history-loading">
      ${t("common.loading")}
    </div>`;
	return b`
    <div class="dashboard-history__preview-wrap">
      <div class="dashboard-history__section-title">${t("dashboard.history.previewTitle")}</div>
      ${renderHistoryPreview(snapshot, state.activeSlug, version)}
    </div>
    <div class="dashboard-history__diff">
      <div class="dashboard-history__section-title">${t("dashboard.history.diffTitle")}</div>
      ${current ? renderHistoryDiff(snapshot, current) : A}
    </div>
    <div class="dashboard-history__restore">
      ${isNewest ? history.confirmRestore ? b`
                <span class="dashboard-history__confirm"
                  >${t("dashboard.history.restoreConfirm")}</span
                >
                <button
                  class="bs-btn bs-btn--small bs-btn--primary"
                  type="button"
                  ?disabled=${history.restoring}
                  data-test-id="dashboard-history-restore-confirm"
                  @click=${async () => {
		history.restoring = true;
		props.onRequestUpdate?.();
		await undoWorkspace(state, props.transport);
		history.restoring = false;
		history.confirmRestore = false;
		closeHistory(props, viewState);
	}}
                >
                  ${t("dashboard.history.restore")}
                </button>
                <button
                  class="bs-btn bs-btn--small"
                  type="button"
                  @click=${() => {
		history.confirmRestore = false;
		props.onRequestUpdate?.();
	}}
                >
                  ${t("common.cancel")}
                </button>
              ` : b`<button
                class="bs-btn bs-btn--small"
                type="button"
                data-test-id="dashboard-history-restore"
                @click=${() => {
		history.confirmRestore = true;
		props.onRequestUpdate?.();
	}}
              >
                ${t("dashboard.history.restore")}
              </button>` : b`<span class="card-sub">${t("dashboard.history.restoreOnlyNewest")}</span>`}
    </div>
  `;
}
/**
* Category mini-glyphs for the snapshot preview cells. The preview resolves NO
* live data (SPEC: previews stay cheap), so each cell shows a static hint of the
* widget's KIND — a faux sparkline for charts, stacked rows for tables/lists, a
* bold value bar for stat cards — so a past layout reads as intentional rather
* than a row of empty boxes. Drawn like `icons.ts`: bare 24×24 stroke paths.
*/
const HISTORY_PREVIEW_GLYPHS = {
	chart: w`<polyline points="3 15 8 10 12 13 17 6 21 9" /><path d="M3 20h18" opacity="0.5" />`,
	"stat-card": w`<path d="M4 8h9" stroke-width="2.6" /><path d="M4 14h6" opacity="0.6" />`,
	table: w`<rect x="3" y="5" width="18" height="14" rx="1.5" /><path d="M3 10h18M3 15h18M9 5v14" opacity="0.6" />`,
	markdown: w`<path d="M4 7h16M4 12h16M4 17h9" opacity="0.85" />`,
	notes: w`<path d="M5 6h11M5 11h11M5 16h7" opacity="0.8" /><path d="M16 15l3-3 2 2-3 3-2 1z" />`,
	list: w`<circle cx="5" cy="7" r="1" /><circle cx="5" cy="12" r="1" /><circle cx="5" cy="17" r="1" /><path d="M9 7h11M9 12h11M9 17h7" opacity="0.8" />`,
	gauge: w`<path d="M4 16a8 8 0 0 1 16 0" /><path d="M12 16l4-3" />`,
	button: w`<rect x="4" y="9" width="16" height="6" rx="3" />`,
	frame: w`<rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 8h18" opacity="0.6" />`,
	custom: w`<path
    d="M4 7h3a1.5 1.5 0 1 0 3 0h3v3a1.5 1.5 0 1 1 0 3v3h-3a1.5 1.5 0 1 0-3 0H4v-3a1.5 1.5 0 1 1 0-3z"
  />`,
	default: w`<rect x="4" y="5" width="16" height="14" rx="2" opacity="0.6" />`
};
const HISTORY_PREVIEW_GLYPH_ALIASES = {
	activity: "list",
	"agent-status": "list",
	approvals: "list",
	sessions: "list",
	instances: "list",
	cron: "list",
	chat: "list",
	usage: "gauge",
	"action-button": "button",
	"action-form": "button",
	"iframe-embed": "frame",
	preview: "frame"
};
/** Resolve a widget kind (`builtin:chart`, `custom:foo`) to its preview glyph. */
function historyPreviewGlyph(kind) {
	const base = kind.startsWith("custom:") ? "custom" : kind.replace(/^builtin:/, "");
	return b`<svg
    class="dashboard-history__cell-glyph"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.6"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    ${HISTORY_PREVIEW_GLYPHS[base] ?? HISTORY_PREVIEW_GLYPHS[HISTORY_PREVIEW_GLYPH_ALIASES[base] ?? "default"]}
  </svg>`;
}
/**
* Static read-only preview of a snapshot's active tab: reuses the grid placement
* math but strips every interaction (no drag/resize handles, menus, or live
* bindings) so a past state renders without any write affordance. Each cell carries
* a per-kind glyph (no data resolved) and the grid is captioned with its version.
*/
function renderHistoryPreview(snapshot, activeSlug, version) {
	const tab = (activeSlug ? snapshot.tabs.find((entry) => entry.slug === activeSlug) : void 0) ?? visibleTabs(snapshot)[0] ?? snapshot.tabs[0];
	if (!tab || tab.widgets.length === 0) return b`<div class="dashboard-history__preview dashboard-history__preview--empty">
      ${t("dashboard.history.previewEmpty")}
    </div>`;
	const rows = gridRowCount(tab.widgets);
	return b`
    <div
      class="dashboard-history__preview dashboard-grid dashboard-grid--readonly"
      style="min-height: ${rows * 56 + Math.max(0, rows - 1) * 12}px"
      data-test-id="dashboard-history-preview"
      aria-hidden="true"
    >
      ${tab.widgets.map((widget) => {
		const agent = dashboardAgentProvenance(widget.createdBy);
		return b`
          <div class="dashboard-history__cell" style=${gridPlacementStyle(widget.grid)}>
            ${historyPreviewGlyph(widget.kind)}
            <span class="dashboard-history__cell-title">${widget.title || widget.kind}</span>
            ${agent ? b`<span class="dashboard-widget__provenance"
                    >${t("dashboard.widget.provenanceChip")}</span
                  >` : A}
          </div>
        `;
	})}
    </div>
    <div class="dashboard-history__preview-caption">
      ${t("dashboard.history.previewCaption", { version: String(version) })}
    </div>
  `;
}
/** Compact changelist (added/removed/moved/retitled) grouped by author (m2). */
function renderHistoryDiff(snapshot, current) {
	const diff = computeWorkspaceDiff(snapshot, current);
	if (diff.length === 0) return b`<div class="card-sub" data-test-id="dashboard-history-diff-empty">
      ${t("dashboard.history.diffEmpty")}
    </div>`;
	return b`
    <div class="dashboard-history__diff-groups" data-test-id="dashboard-history-diff">
      ${groupDiffByActor(diff).map((group) => b`
          <div class="dashboard-history__diff-group">
            <div class="dashboard-history__diff-actor">
              ${group.actor ?? t("dashboard.history.actorUnknown")}
            </div>
            <ul class="dashboard-history__diff-list">
              ${group.entries.map((entry) => b`
                  <li class="dashboard-history__diff-item">
                    <span class="dashboard-history__diff-kind"
                      >${t(`dashboard.history.kind.${entry.kind}`)}</span
                    >
                    <span class="dashboard-history__diff-label">${entry.label}</span>
                    ${entry.detail ? b`<span class="dashboard-history__diff-detail">${entry.detail}</span>` : A}
                  </li>
                `)}
            </ul>
          </div>
        `)}
    </div>
  `;
}
/**
* Widget-gallery dialog (w3): browse an operator-entered registry index, then
* install a bundle. SECURITY — the browse/fetch happens CLIENT-SIDE (the operator's
* browser); the host only receives already-fetched bytes and writes a `pending`
* widget behind the approval gate. The requested capabilities are surfaced BEFORE
* the operator installs (and therefore before they approve).
*/
function renderGalleryDialog(props, state, viewState) {
	const gallery = viewState.gallery;
	if (!gallery) return A;
	const requestUpdate = () => props.onRequestUpdate?.();
	const close = () => {
		viewState.gallery = null;
		requestUpdate();
	};
	const onUrlInput = (event) => {
		gallery.indexUrl = event.currentTarget.value;
	};
	const setMode = (mode) => {
		gallery.mode = mode;
		gallery.selected = null;
		gallery.selectedRecipe = null;
		gallery.error = null;
		requestUpdate();
	};
	const browse = async () => {
		const url = gallery.indexUrl.trim();
		if (!url) return;
		gallery.busy = true;
		gallery.error = null;
		gallery.selected = null;
		gallery.selectedRecipe = null;
		requestUpdate();
		try {
			const [entries, recipes] = await Promise.all([fetchGalleryIndex(url), fetchGalleryRecipes(url)]);
			gallery.entries = entries;
			gallery.recipes = recipes;
			persistGalleryUrl(props.storage, url);
		} catch (err) {
			gallery.error = formatGalleryError(err);
		} finally {
			gallery.busy = false;
			requestUpdate();
		}
	};
	const preview = async (entry) => {
		gallery.busy = true;
		gallery.error = null;
		requestUpdate();
		try {
			gallery.selected = await fetchWidgetBundle(entry.manifestUrl);
		} catch (err) {
			gallery.error = formatGalleryError(err);
		} finally {
			gallery.busy = false;
			requestUpdate();
		}
	};
	const previewRecipe = async (entry) => {
		gallery.busy = true;
		gallery.error = null;
		requestUpdate();
		try {
			gallery.selectedRecipe = await fetchRecipe(entry.manifestUrl);
		} catch (err) {
			gallery.error = formatGalleryError(err);
		} finally {
			gallery.busy = false;
			requestUpdate();
		}
	};
	const installRecipeFlow = async () => {
		const recipe = gallery.selectedRecipe;
		if (!recipe) return;
		gallery.busy = true;
		gallery.error = null;
		requestUpdate();
		try {
			if (!await installRecipe(state, props.transport, recipe)) {
				gallery.error = state.actionError ?? formatGalleryError(/* @__PURE__ */ new Error("Install failed."));
				gallery.busy = false;
				requestUpdate();
				return;
			}
			const firstSlug = recipe.doc.tabs[0]?.slug;
			if (firstSlug) {
				state.activeSlug = firstSlug;
				props.onNavigate?.(firstSlug);
			}
			viewState.gallery = null;
			requestUpdate();
		} catch (err) {
			gallery.error = formatGalleryError(err);
			gallery.busy = false;
			requestUpdate();
		}
	};
	const install = async () => {
		const bundle = gallery.selected;
		if (!bundle) return;
		gallery.busy = true;
		gallery.error = null;
		requestUpdate();
		try {
			await installGalleryWidget(props.transport, bundle);
			const activeTab = state.workspace ? findTab(state.workspace, state.activeSlug) : void 0;
			if (props.transport && activeTab) await props.transport.request("dashboard.widget.add", {
				tab: activeTab.slug,
				widget: {
					kind: `custom:${bundle.name}`,
					title: bundle.title,
					grid: installPlacementGrid(activeTab, bundle)
				}
			});
			await loadWorkspace(state, props.transport, { silent: true });
			viewState.gallery = null;
			requestUpdate();
		} catch (err) {
			gallery.error = formatGalleryError(err);
			gallery.busy = false;
			requestUpdate();
		}
	};
	const renderWidgetsTab = () => gallery.selected ? renderGalleryDetail(gallery.selected, () => {
		gallery.selected = null;
		requestUpdate();
	}, () => void install(), gallery.busy) : renderGalleryList(gallery, (entry) => void preview(entry));
	const renderTemplatesTab = () => gallery.selectedRecipe ? renderRecipeDetail(gallery.selectedRecipe, () => {
		gallery.selectedRecipe = null;
		requestUpdate();
	}, () => void installRecipeFlow(), gallery.busy) : renderRecipeList(gallery, (entry) => void previewRecipe(entry));
	return renderModal(t("dashboard.gallery.title"), close, b`
      <div class="dashboard-gallery" data-test-id="dashboard-gallery">
        <div class="dashboard-gallery__header">
          <div class="card-title">${t("dashboard.gallery.title")}</div>
          <div class="card-sub">${t("dashboard.gallery.subtitle")}</div>
        </div>
        <div class="dashboard-gallery__tabs" role="tablist">
          <button
            class="dashboard-gallery__tab ${gallery.mode === "widgets" ? "is-active" : ""}"
            type="button"
            role="tab"
            aria-selected=${gallery.mode === "widgets"}
            data-test-id="dashboard-gallery-tab-widgets"
            @click=${() => setMode("widgets")}
          >
            ${t("dashboard.gallery.tabWidgets")}
          </button>
          <button
            class="dashboard-gallery__tab ${gallery.mode === "templates" ? "is-active" : ""}"
            type="button"
            role="tab"
            aria-selected=${gallery.mode === "templates"}
            data-test-id="dashboard-gallery-tab-templates"
            @click=${() => setMode("templates")}
          >
            ${t("dashboard.gallery.tabTemplates")}
          </button>
        </div>
        <div class="dashboard-gallery__browse">
          <input
            class="bs-dialog__input"
            type="url"
            inputmode="url"
            data-test-id="dashboard-gallery-url"
            placeholder=${t("dashboard.gallery.urlPlaceholder")}
            aria-label=${t("dashboard.gallery.urlLabel")}
            .value=${gallery.indexUrl}
            @input=${onUrlInput}
          />
          <button
            class="bs-btn bs-btn--small bs-btn--primary"
            type="button"
            data-test-id="dashboard-gallery-browse"
            ?disabled=${gallery.busy}
            @click=${() => void browse()}
          >
            ${t("dashboard.gallery.browse")}
          </button>
        </div>
        ${gallery.error ? b`<div class="callout danger" role="alert" data-test-id="dashboard-gallery-error">
                ${gallery.error}
              </div>` : A}
        ${gallery.mode === "templates" ? renderTemplatesTab() : renderWidgetsTab()}
      </div>
    `);
}
/** Browse results: one row per registry entry. */
function renderGalleryList(gallery, onSelect) {
	if (gallery.entries === null) return A;
	if (gallery.entries.length === 0) return b`<div class="dashboard-gallery__empty">${t("dashboard.gallery.empty")}</div>`;
	return b`
    <ul class="dashboard-gallery__list" data-test-id="dashboard-gallery-list">
      ${gallery.entries.map((entry) => b`
          <li class="dashboard-gallery__item">
            <div class="dashboard-gallery__item-body">
              <div class="dashboard-gallery__item-name">${entry.name}</div>
              ${entry.description ? b`<div class="dashboard-gallery__item-desc">${entry.description}</div>` : A}
            </div>
            <button
              class="bs-btn bs-btn--small"
              type="button"
              data-test-id="dashboard-gallery-select"
              ?disabled=${gallery.busy}
              @click=${() => onSelect(entry)}
            >
              ${t("dashboard.gallery.view")}
            </button>
          </li>
        `)}
    </ul>
  `;
}
/** Selected-bundle detail: surfaces the REQUESTED CAPABILITIES before installing. */
function renderGalleryDetail(bundle, onBack, onInstall, busy) {
	return b`
    <div class="dashboard-gallery__detail" data-test-id="dashboard-gallery-detail">
      <div class="dashboard-gallery__item-name">${bundle.title}</div>
      <div class="dashboard-gallery__caps">
        <div class="dashboard-gallery__caps-label">${t("dashboard.gallery.capabilities")}</div>
        ${bundle.capabilities.length === 0 ? b`<span class="dashboard-gallery__cap"
                >${t("dashboard.gallery.noCapabilities")}</span
              >` : bundle.capabilities.map((cap) => b`<span class="dashboard-gallery__cap" data-test-id="dashboard-gallery-cap"
                    >${cap}</span
                  >`)}
      </div>
      <div class="dashboard-gallery__pending-note">${t("dashboard.gallery.pendingNote")}</div>
      <div class="bs-dialog__actions">
        <button
          class="bs-btn bs-btn--primary"
          type="button"
          data-test-id="dashboard-gallery-install"
          ?disabled=${busy}
          @click=${onInstall}
        >
          ${t("dashboard.gallery.install")}
        </button>
        <button class="bs-btn" type="button" @click=${onBack}>${t("common.back")}</button>
      </div>
    </div>
  `;
}
/** Templates tab list (#60): one row per recipe, with its "what it needs" connector hint. */
function renderRecipeList(gallery, onSelect) {
	if (gallery.recipes === null) return A;
	if (gallery.recipes.length === 0) return b`<div class="dashboard-gallery__empty">${t("dashboard.gallery.recipesEmpty")}</div>`;
	return b`
    <ul class="dashboard-gallery__list" data-test-id="dashboard-gallery-recipe-list">
      ${gallery.recipes.map((entry) => b`
          <li class="dashboard-gallery__item">
            <div class="dashboard-gallery__item-body">
              <div class="dashboard-gallery__item-name">${entry.title}</div>
              ${entry.description ? b`<div class="dashboard-gallery__item-desc">${entry.description}</div>` : A}
              <div class="dashboard-gallery__recipe-needs">
                ${entry.connectors.length === 0 ? t("dashboard.gallery.recipeNeedsNothing") : t("dashboard.gallery.recipeNeedsConnectors", { connectors: entry.connectors.join(", ") })}
              </div>
            </div>
            <button
              class="bs-btn bs-btn--small"
              type="button"
              data-test-id="dashboard-gallery-recipe-select"
              ?disabled=${gallery.busy}
              @click=${() => onSelect(entry)}
            >
              ${t("dashboard.gallery.view")}
            </button>
          </li>
        `)}
    </ul>
  `;
}
/**
* Selected-recipe detail (#60): an HONEST "what this board will ask for" — the grant
* list, per connector, with a human label for each tool — surfaced BEFORE install. On
* install the grants land `requested`; the approvals widget shows the pending cards.
*/
function renderRecipeDetail(recipe, onBack, onInstall, busy) {
	const connectors = Object.entries(recipe.grantsManifest);
	return b`
    <div class="dashboard-gallery__detail" data-test-id="dashboard-gallery-recipe-detail">
      <div class="dashboard-gallery__item-name">${recipe.title}</div>
      <div class="dashboard-gallery__item-desc">${recipe.description}</div>
      <div class="dashboard-gallery__recipe-grants">
        <div class="dashboard-gallery__caps-label">${t("dashboard.gallery.recipeNeedsLabel")}</div>
        ${connectors.length === 0 ? b`<div class="dashboard-gallery__recipe-nogrants">
                ${t("dashboard.gallery.recipeNoGrants")}
              </div>` : connectors.map(([, grant]) => b`
                  <div class="dashboard-gallery__recipe-connector">
                    <div class="dashboard-gallery__recipe-connector-name">${grant.label}</div>
                    ${grant.reason ? b`<div class="dashboard-gallery__recipe-connector-reason">
                            ${grant.reason}
                          </div>` : A}
                    <ul class="dashboard-gallery__recipe-tools">
                      ${(grant.tools ?? []).map((tool) => b`
                          <li
                            class="dashboard-gallery__recipe-tool"
                            data-test-id="dashboard-gallery-recipe-tool"
                          >
                            <code>${tool.id}</code>
                            <span>${tool.label}</span>
                            ${tool.readOnly ? b`<span class="dashboard-gallery__recipe-readonly"
                                    >${t("dashboard.gallery.recipeReadOnly")}</span
                                  >` : A}
                          </li>
                        `)}
                    </ul>
                  </div>
                `)}
      </div>
      <div class="dashboard-gallery__pending-note">${t("dashboard.gallery.recipeInstallNote")}</div>
      <div class="bs-dialog__actions">
        <button
          class="bs-btn bs-btn--primary"
          type="button"
          data-test-id="dashboard-gallery-recipe-install"
          ?disabled=${busy}
          @click=${onInstall}
        >
          ${t("dashboard.gallery.recipeInstall")}
        </button>
        <button class="bs-btn" type="button" @click=${onBack}>${t("common.back")}</button>
      </div>
    </div>
  `;
}
/**
* `<boardstate-view>` — the reference view custom element. Renders into light DOM
* (so injected theme tokens / CSS cascade). Set `transport` + `connected` to drive
* it; `strings`/`onNavigate`/`storage`/`confirm`/`embed`/`basePath`/`initialTab`
* customize behavior.
*/
var BoardstateViewElement = class extends i$2 {
	constructor(..._args) {
		super(..._args);
		this.transport = null;
		this.connected = false;
		this.operator = false;
	}
	createRenderRoot() {
		return this;
	}
	static {
		this.properties = {
			transport: { attribute: false },
			connected: { type: Boolean },
			strings: { attribute: false },
			onNavigate: { attribute: false },
			storage: { attribute: false },
			confirm: { attribute: false },
			embed: { attribute: false },
			basePath: { type: String },
			initialTab: { type: String },
			sessionKey: { type: String },
			logbookHref: { type: String },
			operator: { type: Boolean }
		};
	}
	render() {
		return renderBoardstateView({
			host: this,
			transport: this.transport,
			connected: this.connected,
			onRequestUpdate: () => this.requestUpdate(),
			...this.strings ? { strings: this.strings } : {},
			...this.onNavigate ? { onNavigate: this.onNavigate } : {},
			...this.storage ? { storage: this.storage } : {},
			...this.confirm ? { confirm: this.confirm } : {},
			...this.embed ? { embed: this.embed } : {},
			...this.basePath !== void 0 ? { basePath: this.basePath } : {},
			...this.initialTab !== void 0 ? { initialTab: this.initialTab } : {},
			...this.sessionKey !== void 0 ? { sessionKey: this.sessionKey } : {},
			...this.logbookHref !== void 0 ? { logbookHref: this.logbookHref } : {},
			operator: this.operator
		});
	}
	disconnectedCallback() {
		super.disconnectedCallback();
		stopDashboard(this);
		stopBoardstateView(this);
	}
};
if (typeof customElements !== "undefined" && !customElements.get("boardstate-view")) customElements.define("boardstate-view", BoardstateViewElement);
//#endregion
//#region src/boardstate-header.ts
var BoardstateHeaderElement = class extends i$2 {
	constructor(..._args) {
		super(..._args);
		this.currentLabel = "";
		this.agentLabel = "";
		this.brandLabel = "";
		this.overviewHref = "";
		this.handleOverviewClick = (event) => {
			if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
			event.preventDefault();
			this.dispatchEvent(new CustomEvent("navigate", {
				detail: "overview",
				bubbles: true,
				composed: true
			}));
		};
	}
	/** Render into light DOM so app CSS/theme tokens apply. */
	createRenderRoot() {
		return this;
	}
	static {
		this.properties = {
			currentLabel: { type: String },
			agentLabel: { type: String },
			brandLabel: { type: String },
			overviewHref: { type: String }
		};
	}
	render() {
		const label = this.currentLabel.trim();
		const agentLabel = this.agentLabel.trim();
		const brand = this.brandLabel.trim();
		return b`
      <div class="dashboard-header">
        <div class="dashboard-header__breadcrumb">
          ${brand ? this.overviewHref ? b`<a
                    class="dashboard-header__breadcrumb-link"
                    href=${this.overviewHref}
                    @click=${this.handleOverviewClick}
                    >${brand}</a
                  >` : b`<span class="dashboard-header__breadcrumb-link">${brand}</span>` : A}
          ${agentLabel ? b`
                  <span class="dashboard-header__breadcrumb-segment">
                    ${brand ? b`<span class="dashboard-header__breadcrumb-sep">›</span>` : A}
                    <span class="dashboard-header__breadcrumb-context" title=${agentLabel}>
                      ${agentLabel}
                    </span>
                  </span>
                ` : A}
          ${label ? b`
                  ${brand || agentLabel ? b`<span class="dashboard-header__breadcrumb-sep">›</span>` : A}
                  <span class="dashboard-header__breadcrumb-current">${label}</span>
                ` : A}
        </div>
        <div class="dashboard-header__actions">
          <slot></slot>
        </div>
      </div>
    `;
	}
};
if (typeof customElements !== "undefined" && !customElements.get("boardstate-header")) customElements.define("boardstate-header", BoardstateHeaderElement);
//#endregion
export { BUILTIN_WIDGET_RENDERERS, BoardstateHeaderElement, BoardstateViewElement, en as DEFAULT_STRINGS, boardstateDataVersion, t as boardstateString, bumpBoardstateDataVersion, displayWidgetTitle, getBuiltinRenderer, icons, renderBoardstateView, renderBuiltinWidget, renderCustomWidget, renderCustomWidgetHost, renderWidgetBody, renderWidgetCell, setBoardstateStrings, stopBoardstateView, toSanitizedMarkdownHtml };
