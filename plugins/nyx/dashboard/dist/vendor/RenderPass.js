import {
	Color
} from './three.module.min.js';
import { Pass } from './Pass.js';

/**
 * This class represents a render pass. It takes a camera and a scene and produces
 * a beauty pass for subsequent post processing effects.
 *
 * ```js
 * const renderPass = new RenderPass( scene, camera );
 * composer.addPass( renderPass );
 * ```
 *
 * @augments Pass
 * @three_import import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
 */
class RenderPass extends Pass {

	/**
	 * Constructs a new render pass.
	 *
	 * @param {Scene} scene - The scene to render.
	 * @param {Camera} camera - The camera.
	 * @param {?Material} [overrideMaterial=null] - The override material. If set, this material is used
	 * for all objects in the scene.
	 * @param {?(number|Color|string)} [clearColor=null] - The clear color of the render pass.
	 * @param {?number} [clearAlpha=null] - The clear alpha of the render pass.
	 */
	constructor( scene, camera, overrideMaterial = null, clearColor = null, clearAlpha = null ) {

		super();

		/**
		 * The scene to render.
		 *
		 * @type {Scene}
		 */
		this.scene = scene;

		/**
		 * The camera.
		 *
		 * @type {Camera}
		 */
		this.camera = camera;

		/**
		 * The override material. If set, this material is used
		 * for all objects in the scene.
		 *
		 * @type {?Material}
		 * @default null
		 */
		this.overrideMaterial = overrideMaterial;

		/**
		 * The clear color of the render pass.
		 *
		 * @type {?(number|Color|string)}
		 * @default null
		 */
		this.clearColor = clearColor;

		/**
		 * The clear alpha of the render pass.
		 *
		 * @type {?number}
		 * @default null
		 */
		this.clearAlpha = clearAlpha;

		/**
		 * Overwritten to perform a clear operation by default.
		 *
		 * @type {boolean}
		 * @default true
		 */
		this.clear = true;

		/**
		 * If set to `true`, only the depth can be cleared when `clear` is to `false`.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.clearDepth = false;

		/**
		 * Overwritten to disable the swap.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.needsSwap = false;
		this._oldClearColor = new Color();

	}

	/**
	 * Performs a beauty pass with the configured scene and camera.
	 *
	 * @param {WebGLRenderer} renderer - The renderer.
	 * @param {WebGLRenderTarget} writeBuffer - The write buffer. This buffer is intended as the rendering
	 * destination for the pass.
	 * @param {WebGLRenderTarget} readBuffer - The read buffer. The pass can access the result from the
	 * previous pass from this buffer.
	 * @param {number} deltaTime - The delta time in seconds.
	 * @param {boolean} maskActive - Whether masking is active or not.
	 */
	render( renderer, writeBuffer, readBuffer /*, deltaTime, maskActive */ ) {

		const oldAutoClear = renderer.autoClear;
		renderer.autoClear = false;

		let oldClearAlpha, oldOverrideMaterial;

		if ( this.overrideMaterial !== null ) {

			oldOverrideMaterial = this.scene.overrideMaterial;

			this.scene.overrideMaterial = this.overrideMaterial;

		}

		if ( this.clearColor !== null ) {

			renderer.getClearColor( this._oldClearColor );
			renderer.setClearColor( this.clearColor, renderer.getClearAlpha() );

		}

		if ( this.clearAlpha !== null ) {

			oldClearAlpha = renderer.getClearAlpha();
			renderer.setClearAlpha( this.clearAlpha );

		}

		if ( this.clearDepth == true ) {

			renderer.clearDepth();

		}

		renderer.setRenderTarget( this.renderToScreen ? null : readBuffer );

		if ( this.clear === true ) {

			// TODO: Avoid using autoClear properties, see https://github.com/mrdoob/three.js/pull/15571#issuecomment-465669600
			renderer.clear( renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil );

		}

		renderer.render( this.scene, this.camera );

		// restore

		if ( this.clearColor !== null ) {

			renderer.setClearColor( this._oldClearColor );

		}

		if ( this.clearAlpha !== null ) {

			renderer.setClearAlpha( oldClearAlpha );

		}

		if ( this.overrideMaterial !== null ) {

			this.scene.overrideMaterial = oldOverrideMaterial;

		}

		renderer.autoClear = oldAutoClear;

	}

}

export { RenderPass };
