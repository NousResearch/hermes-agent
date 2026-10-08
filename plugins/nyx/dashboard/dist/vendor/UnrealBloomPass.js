import {
	AdditiveBlending,
	Color,
	HalfFloatType,
	MeshBasicMaterial,
	ShaderMaterial,
	UniformsUtils,
	Vector2,
	Vector3,
	WebGLRenderTarget
} from './three.module.min.js';
import { Pass, FullScreenQuad } from './Pass.js';
import { CopyShader } from './CopyShader.js';
import { LuminosityHighPassShader } from './LuminosityHighPassShader.js';

/**
 * This pass is inspired by the bloom pass of Unreal Engine. It creates a
 * mip map chain of bloom textures and blurs them with different radii. Because
 * of the weighted combination of mips, and because larger blurs are done on
 * higher mips, this effect provides good quality and performance.
 *
 * When using this pass, tone mapping must be enabled in the renderer settings.
 *
 * Reference:
 * - [Bloom in Unreal Engine](https://docs.unrealengine.com/latest/INT/Engine/Rendering/PostProcessEffects/Bloom/)
 *
 * ```js
 * const resolution = new THREE.Vector2( window.innerWidth, window.innerHeight );
 * const bloomPass = new UnrealBloomPass( resolution, 1.5, 0.4, 0.85 );
 * composer.addPass( bloomPass );
 * ```
 *
 * @augments Pass
 * @three_import import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
 */
class UnrealBloomPass extends Pass {

	/**
	 * Constructs a new Unreal Bloom pass.
	 *
	 * @param {Vector2} [resolution] - The effect's resolution.
	 * @param {number} [strength=1] - The Bloom strength.
	 * @param {number} radius - The Bloom radius.
	 * @param {number} threshold - The luminance threshold limits which bright areas contribute to the Bloom effect.
	 */
	constructor( resolution, strength = 1, radius, threshold ) {

		super();

		/**
		 * The Bloom strength.
		 *
		 * @type {number}
		 * @default 1
		 */
		this.strength = strength;

		/**
		 * The Bloom radius.
		 *
		 * @type {number}
		 */
		this.radius = radius;

		/**
		 * The luminance threshold limits which bright areas contribute to the Bloom effect.
		 *
		 * @type {number}
		 */
		this.threshold = threshold;

		/**
		 * The effect's resolution.
		 *
		 * @type {Vector2}
		 * @default (256,256)
		 */
		this.resolution = ( resolution !== undefined ) ? new Vector2( resolution.x, resolution.y ) : new Vector2( 256, 256 );

		/**
		 * The effect's clear color
		 *
		 * @type {Color}
		 * @default (0,0,0)
		 */
		this.clearColor = new Color( 0, 0, 0 );

		/**
		 * Overwritten to disable the swap.
		 *
		 * @type {boolean}
		 * @default false
		 */
		this.needsSwap = false;

		// internals

		// render targets
		this.renderTargetsHorizontal = [];
		this.renderTargetsVertical = [];
		this.nMips = 5;
		let resx = Math.round( this.resolution.x / 2 );
		let resy = Math.round( this.resolution.y / 2 );

		this.renderTargetBright = new WebGLRenderTarget( resx, resy, { type: HalfFloatType } );
		this.renderTargetBright.texture.name = 'UnrealBloomPass.bright';
		this.renderTargetBright.texture.generateMipmaps = false;

		for ( let i = 0; i < this.nMips; i ++ ) {

			const renderTargetHorizontal = new WebGLRenderTarget( resx, resy, { type: HalfFloatType } );

			renderTargetHorizontal.texture.name = 'UnrealBloomPass.h' + i;
			renderTargetHorizontal.texture.generateMipmaps = false;

			this.renderTargetsHorizontal.push( renderTargetHorizontal );

			const renderTargetVertical = new WebGLRenderTarget( resx, resy, { type: HalfFloatType } );

			renderTargetVertical.texture.name = 'UnrealBloomPass.v' + i;
			renderTargetVertical.texture.generateMipmaps = false;

			this.renderTargetsVertical.push( renderTargetVertical );

			resx = Math.round( resx / 2 );

			resy = Math.round( resy / 2 );

		}

		// luminosity high pass material

		const highPassShader = LuminosityHighPassShader;
		this.highPassUniforms = UniformsUtils.clone( highPassShader.uniforms );

		this.highPassUniforms[ 'luminosityThreshold' ].value = threshold;
		this.highPassUniforms[ 'smoothWidth' ].value = 0.01;

		this.materialHighPassFilter = new ShaderMaterial( {
			uniforms: this.highPassUniforms,
			vertexShader: highPassShader.vertexShader,
			fragmentShader: highPassShader.fragmentShader
		} );

		// gaussian blur materials

		this.separableBlurMaterials = [];
		// These sizes have been changed to account for the altered coefficients-calculation to avoid blockiness,
		// while retaining the same blur-strength. For details see https://github.com/mrdoob/three.js/pull/31528
		const kernelSizeArray = [ 6, 10, 14, 18, 22 ];
		resx = Math.round( this.resolution.x / 2 );
		resy = Math.round( this.resolution.y / 2 );

		for ( let i = 0; i < this.nMips; i ++ ) {

			this.separableBlurMaterials.push( this._getSeparableBlurMaterial( kernelSizeArray[ i ] ) );

			this.separableBlurMaterials[ i ].uniforms[ 'invSize' ].value = new Vector2( 1 / resx, 1 / resy );

			resx = Math.round( resx / 2 );

			resy = Math.round( resy / 2 );

		}

		// composite material

		this.compositeMaterial = this._getCompositeMaterial( this.nMips );
		this.compositeMaterial.uniforms[ 'blurTexture1' ].value = this.renderTargetsVertical[ 0 ].texture;
		this.compositeMaterial.uniforms[ 'blurTexture2' ].value = this.renderTargetsVertical[ 1 ].texture;
		this.compositeMaterial.uniforms[ 'blurTexture3' ].value = this.renderTargetsVertical[ 2 ].texture;
		this.compositeMaterial.uniforms[ 'blurTexture4' ].value = this.renderTargetsVertical[ 3 ].texture;
		this.compositeMaterial.uniforms[ 'blurTexture5' ].value = this.renderTargetsVertical[ 4 ].texture;
		this.compositeMaterial.uniforms[ 'bloomStrength' ].value = strength;
		this.compositeMaterial.uniforms[ 'bloomRadius' ].value = 0.1;

		const bloomFactors = [ 1.0, 0.8, 0.6, 0.4, 0.2 ];
		this.compositeMaterial.uniforms[ 'bloomFactors' ].value = bloomFactors;
		this.bloomTintColors = [ new Vector3( 1, 1, 1 ), new Vector3( 1, 1, 1 ), new Vector3( 1, 1, 1 ), new Vector3( 1, 1, 1 ), new Vector3( 1, 1, 1 ) ];
		this.compositeMaterial.uniforms[ 'bloomTintColors' ].value = this.bloomTintColors;

		// blend material

		this.copyUniforms = UniformsUtils.clone( CopyShader.uniforms );

		this.blendMaterial = new ShaderMaterial( {
			uniforms: this.copyUniforms,
			vertexShader: CopyShader.vertexShader,
			fragmentShader: CopyShader.fragmentShader,
			blending: AdditiveBlending,
			depthTest: false,
			depthWrite: false,
			transparent: true
		} );

		this._oldClearColor = new Color();
		this._oldClearAlpha = 1;

		this._basic = new MeshBasicMaterial();

		this._fsQuad = new FullScreenQuad( null );

	}

	/**
	 * Frees the GPU-related resources allocated by this instance. Call this
	 * method whenever the pass is no longer used in your app.
	 */
	dispose() {

		for ( let i = 0; i < this.renderTargetsHorizontal.length; i ++ ) {

			this.renderTargetsHorizontal[ i ].dispose();

		}

		for ( let i = 0; i < this.renderTargetsVertical.length; i ++ ) {

			this.renderTargetsVertical[ i ].dispose();

		}

		this.renderTargetBright.dispose();

		//

		for ( let i = 0; i < this.separableBlurMaterials.length; i ++ ) {

			this.separableBlurMaterials[ i ].dispose();

		}

		this.compositeMaterial.dispose();
		this.blendMaterial.dispose();
		this._basic.dispose();

		//

		this._fsQuad.dispose();

	}

	/**
	 * Sets the size of the pass.
	 *
	 * @param {number} width - The width to set.
	 * @param {number} height - The height to set.
	 */
	setSize( width, height ) {

		let resx = Math.round( width / 2 );
		let resy = Math.round( height / 2 );

		this.renderTargetBright.setSize( resx, resy );

		for ( let i = 0; i < this.nMips; i ++ ) {

			this.renderTargetsHorizontal[ i ].setSize( resx, resy );
			this.renderTargetsVertical[ i ].setSize( resx, resy );

			this.separableBlurMaterials[ i ].uniforms[ 'invSize' ].value = new Vector2( 1 / resx, 1 / resy );

			resx = Math.round( resx / 2 );
			resy = Math.round( resy / 2 );

		}

	}

	/**
	 * Performs the Bloom pass.
	 *
	 * @param {WebGLRenderer} renderer - The renderer.
	 * @param {WebGLRenderTarget} writeBuffer - The write buffer. This buffer is intended as the rendering
	 * destination for the pass.
	 * @param {WebGLRenderTarget} readBuffer - The read buffer. The pass can access the result from the
	 * previous pass from this buffer.
	 * @param {number} deltaTime - The delta time in seconds.
	 * @param {boolean} maskActive - Whether masking is active or not.
	 */
	render( renderer, writeBuffer, readBuffer, deltaTime, maskActive ) {

		renderer.getClearColor( this._oldClearColor );
		this._oldClearAlpha = renderer.getClearAlpha();
		const oldAutoClear = renderer.autoClear;
		renderer.autoClear = false;

		renderer.setClearColor( this.clearColor, 0 );

		if ( maskActive ) renderer.state.buffers.stencil.setTest( false );

		// Render input to screen

		if ( this.renderToScreen ) {

			this._fsQuad.material = this._basic;
			this._basic.map = readBuffer.texture;

			renderer.setRenderTarget( null );
			renderer.clear();
			this._fsQuad.render( renderer );

		}

		// 1. Extract Bright Areas

		this.highPassUniforms[ 'tDiffuse' ].value = readBuffer.texture;
		this.highPassUniforms[ 'luminosityThreshold' ].value = this.threshold;
		this._fsQuad.material = this.materialHighPassFilter;

		renderer.setRenderTarget( this.renderTargetBright );
		renderer.clear();
		this._fsQuad.render( renderer );

		// 2. Blur All the mips progressively

		let inputRenderTarget = this.renderTargetBright;

		for ( let i = 0; i < this.nMips; i ++ ) {

			this._fsQuad.material = this.separableBlurMaterials[ i ];

			this.separableBlurMaterials[ i ].uniforms[ 'colorTexture' ].value = inputRenderTarget.texture;
			this.separableBlurMaterials[ i ].uniforms[ 'direction' ].value = UnrealBloomPass.BlurDirectionX;
			renderer.setRenderTarget( this.renderTargetsHorizontal[ i ] );
			renderer.clear();
			this._fsQuad.render( renderer );

			this.separableBlurMaterials[ i ].uniforms[ 'colorTexture' ].value = this.renderTargetsHorizontal[ i ].texture;
			this.separableBlurMaterials[ i ].uniforms[ 'direction' ].value = UnrealBloomPass.BlurDirectionY;
			renderer.setRenderTarget( this.renderTargetsVertical[ i ] );
			renderer.clear();
			this._fsQuad.render( renderer );

			inputRenderTarget = this.renderTargetsVertical[ i ];

		}

		// Composite All the mips

		this._fsQuad.material = this.compositeMaterial;
		this.compositeMaterial.uniforms[ 'bloomStrength' ].value = this.strength;
		this.compositeMaterial.uniforms[ 'bloomRadius' ].value = this.radius;
		this.compositeMaterial.uniforms[ 'bloomTintColors' ].value = this.bloomTintColors;

		renderer.setRenderTarget( this.renderTargetsHorizontal[ 0 ] );
		renderer.clear();
		this._fsQuad.render( renderer );

		// Blend it additively over the input texture

		this._fsQuad.material = this.blendMaterial;
		this.copyUniforms[ 'tDiffuse' ].value = this.renderTargetsHorizontal[ 0 ].texture;

		if ( maskActive ) renderer.state.buffers.stencil.setTest( true );

		if ( this.renderToScreen ) {

			renderer.setRenderTarget( null );
			this._fsQuad.render( renderer );

		} else {

			renderer.setRenderTarget( readBuffer );
			this._fsQuad.render( renderer );

		}

		// Restore renderer settings

		renderer.setClearColor( this._oldClearColor, this._oldClearAlpha );
		renderer.autoClear = oldAutoClear;

	}

	// internals

	_getSeparableBlurMaterial( kernelRadius ) {

		const coefficients = [];
		const sigma = kernelRadius / 3;

		for ( let i = 0; i < kernelRadius; i ++ ) {

			coefficients.push( 0.39894 * Math.exp( - 0.5 * i * i / ( sigma * sigma ) ) / sigma );

		}

		return new ShaderMaterial( {

			defines: {
				'KERNEL_RADIUS': kernelRadius
			},

			uniforms: {
				'colorTexture': { value: null },
				'invSize': { value: new Vector2( 0.5, 0.5 ) }, // inverse texture size
				'direction': { value: new Vector2( 0.5, 0.5 ) },
				'gaussianCoefficients': { value: coefficients } // precomputed Gaussian coefficients
			},

			vertexShader:
				`varying vec2 vUv;
				void main() {
					vUv = uv;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
				}`,

			fragmentShader:
				`#include <common>
				varying vec2 vUv;
				uniform sampler2D colorTexture;
				uniform vec2 invSize;
				uniform vec2 direction;
				uniform float gaussianCoefficients[KERNEL_RADIUS];

				void main() {
					float weightSum = gaussianCoefficients[0];
					vec3 diffuseSum = texture2D( colorTexture, vUv ).rgb * weightSum;
					for( int i = 1; i < KERNEL_RADIUS; i ++ ) {
						float x = float(i);
						float w = gaussianCoefficients[i];
						vec2 uvOffset = direction * invSize * x;
						vec3 sample1 = texture2D( colorTexture, vUv + uvOffset ).rgb;
						vec3 sample2 = texture2D( colorTexture, vUv - uvOffset ).rgb;
						diffuseSum += ( sample1 + sample2 ) * w;
					}
					gl_FragColor = vec4( diffuseSum, 1.0 );
				}`
		} );

	}

	_getCompositeMaterial( nMips ) {

		return new ShaderMaterial( {

			defines: {
				'NUM_MIPS': nMips
			},

			uniforms: {
				'blurTexture1': { value: null },
				'blurTexture2': { value: null },
				'blurTexture3': { value: null },
				'blurTexture4': { value: null },
				'blurTexture5': { value: null },
				'bloomStrength': { value: 1.0 },
				'bloomFactors': { value: null },
				'bloomTintColors': { value: null },
				'bloomRadius': { value: 0.0 }
			},

			vertexShader:
				`varying vec2 vUv;
				void main() {
					vUv = uv;
					gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
				}`,

			fragmentShader:
				`varying vec2 vUv;
				uniform sampler2D blurTexture1;
				uniform sampler2D blurTexture2;
				uniform sampler2D blurTexture3;
				uniform sampler2D blurTexture4;
				uniform sampler2D blurTexture5;
				uniform float bloomStrength;
				uniform float bloomRadius;
				uniform float bloomFactors[NUM_MIPS];
				uniform vec3 bloomTintColors[NUM_MIPS];

				float lerpBloomFactor(const in float factor) {
					float mirrorFactor = 1.2 - factor;
					return mix(factor, mirrorFactor, bloomRadius);
				}

				void main() {
					gl_FragColor = bloomStrength * ( lerpBloomFactor(bloomFactors[0]) * vec4(bloomTintColors[0], 1.0) * texture2D(blurTexture1, vUv) +
						lerpBloomFactor(bloomFactors[1]) * vec4(bloomTintColors[1], 1.0) * texture2D(blurTexture2, vUv) +
						lerpBloomFactor(bloomFactors[2]) * vec4(bloomTintColors[2], 1.0) * texture2D(blurTexture3, vUv) +
						lerpBloomFactor(bloomFactors[3]) * vec4(bloomTintColors[3], 1.0) * texture2D(blurTexture4, vUv) +
						lerpBloomFactor(bloomFactors[4]) * vec4(bloomTintColors[4], 1.0) * texture2D(blurTexture5, vUv) );
				}`
		} );

	}

}

UnrealBloomPass.BlurDirectionX = new Vector2( 1.0, 0.0 );
UnrealBloomPass.BlurDirectionY = new Vector2( 0.0, 1.0 );

export { UnrealBloomPass };
