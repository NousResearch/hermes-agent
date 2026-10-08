import { Pass } from './Pass.js';

/**
 * This pass can be used to define a mask during post processing.
 * Meaning only areas of subsequent post processing are affected
 * which lie in the masking area of this pass. Internally, the masking
 * is implemented with the stencil buffer.
 *
 * ```js
 * const maskPass = new MaskPass( scene, camera );
 * composer.addPass( maskPass );
 * ```
 *
 * @augments Pass
 * @three_import import { MaskPass } from 'three/addons/postprocessing/MaskPass.js';
 */
class MaskPass extends Pass {

	/**
	 * Constructs a new mask pass.
	 *
	 * @param {Scene} scene - The 3D objects in this scene will define the mask.
	 * @param {Camera} camera - The camera.
	 */
	constructor( scene, camera ) {

		super();

		/**
		 * The scene that defines the mask.
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
		 * Overwritten to perform a clear operation by default.
		 *
		 * @type {boolean}
		 * @default true
		 */
		this.clear = true;

		/**
		 * Overwritten to disable the swap.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.needsSwap = false;

		/**
		 * Whether to inverse the mask or not.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.inverse = false;

	}

	/**
	 * Performs a mask pass with the configured scene and camera.
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

		const context = renderer.getContext();
		const state = renderer.state;

		// don't update color or depth

		state.buffers.color.setMask( false );
		state.buffers.depth.setMask( false );

		// lock buffers

		state.buffers.color.setLocked( true );
		state.buffers.depth.setLocked( true );

		// set up stencil

		let writeValue, clearValue;

		if ( this.inverse ) {

			writeValue = 0;
			clearValue = 1;

		} else {

			writeValue = 1;
			clearValue = 0;

		}

		state.buffers.stencil.setTest( true );
		state.buffers.stencil.setOp( context.REPLACE, context.REPLACE, context.REPLACE );
		state.buffers.stencil.setFunc( context.ALWAYS, writeValue, 0xffffffff );
		state.buffers.stencil.setClear( clearValue );
		state.buffers.stencil.setLocked( true );

		// draw into the stencil buffer

		renderer.setRenderTarget( readBuffer );
		if ( this.clear ) renderer.clear();
		renderer.render( this.scene, this.camera );

		renderer.setRenderTarget( writeBuffer );
		if ( this.clear ) renderer.clear();
		renderer.render( this.scene, this.camera );

		// unlock color and depth buffer and make them writable for subsequent rendering/clearing

		state.buffers.color.setLocked( false );
		state.buffers.depth.setLocked( false );

		state.buffers.color.setMask( true );
		state.buffers.depth.setMask( true );

		// only render where stencil is set to 1

		state.buffers.stencil.setLocked( false );
		state.buffers.stencil.setFunc( context.EQUAL, 1, 0xffffffff ); // draw if == 1
		state.buffers.stencil.setOp( context.KEEP, context.KEEP, context.KEEP );
		state.buffers.stencil.setLocked( true );

	}

}

/**
 * This pass can be used to clear a mask previously defined with {@link MaskPass}.
 *
 * ```js
 * const clearPass = new ClearMaskPass();
 * composer.addPass( clearPass );
 * ```
 *
 * @augments Pass
 */
class ClearMaskPass extends Pass {

	/**
	 * Constructs a new clear mask pass.
	 */
	constructor() {

		super();

		/**
		 * Overwritten to disable the swap.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.needsSwap = false;

	}

	/**
	 * Performs the clear of the currently defined mask.
	 *
	 * @param {WebGLRenderer} renderer - The renderer.
	 * @param {WebGLRenderTarget} writeBuffer - The write buffer. This buffer is intended as the rendering
	 * destination for the pass.
	 * @param {WebGLRenderTarget} readBuffer - The read buffer. The pass can access the result from the
	 * previous pass from this buffer.
	 * @param {number} deltaTime - The delta time in seconds.
	 * @param {boolean} maskActive - Whether masking is active or not.
	 */
	render( renderer /*, writeBuffer, readBuffer, deltaTime, maskActive */ ) {

		renderer.state.buffers.stencil.setLocked( false );
		renderer.state.buffers.stencil.setTest( false );

	}

}

export { MaskPass, ClearMaskPass };
