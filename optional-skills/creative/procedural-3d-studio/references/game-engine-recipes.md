# Game Engine Integration Recipes

## Godot 4 Integration

1. Copy the generated `.glb` and `.tscn` files into your Godot project's `assets/` directory.
2. Open Godot 4. The engine will automatically import the `.glb` mesh.
3. Instantiate the `.tscn` scene directly into your main world scene:
   ```gdscript
   var tower_scene = preload("res://assets/watchtower.tscn")
   var tower_instance = tower_scene.instantiate()
   add_child(tower_instance)
   ```

## Three.js Web Integration

Load the generated `.glb` asset in Three.js:

```javascript
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.load('models/hero_sword.glb', (gltf) => {
  scene.add(gltf.scene);
});
```

## Unity Integration

Drag and drop the `.glb` or `.obj` into Unity's `Assets/` folder. Unity's model importer automatically maps the PBR diffuse color, normals, and vertex colors.
