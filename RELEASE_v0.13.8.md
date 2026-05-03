# Hermes Agent v0.13.8

This release improves native Android local LiteRT-LM behavior on physical Qualcomm/Adreno phones.

- Requests Android's optional `libOpenCL.so` public native library so LiteRT-LM can initialize the GPU backend when the device exposes OpenCL.
- Checks OpenCL through `System.loadLibrary("OpenCL")` before falling back to direct vendor library paths.
- Keeps CPU fallback for translated or non-OpenCL devices.
- Extends native tool-calling chat requests to the proxy's 300 second generation timeout so longer on-device tasks, such as writing HTML files, are not cut off at the default 120 seconds.
- Keeps focused Android regression coverage for Adreno OpenCL availability and long native tool prompts.
