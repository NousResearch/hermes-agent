// Forward declarations of types of the Python C API.
// Declare them at the same place since redefining typedef is a C11 feature.
// Only use a forward declaration if there is an interdependency between two
// header files.

#ifndef Py_PYTYPEDEFS_H
#define Py_PYTYPEDEFS_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct PyModuleDef PyModuleDef;
typedef struct PyModuleDef_Slot PyModuleDef_Slot;
typedef struct PyMethodDef PyMethodDef;
typedef struct PyGetSetDef PyGetSetDef;
typedef struct PyMemberDef PyMemberDef;

typedef struct _object PyObject;
typedef struct _longobject PyLongObject;
typedef struct _typeobject PyTypeObject;
typedef struct PyCodeObject PyCodeObject;
typedef struct _frame PyFrameObject;

typedef struct _ts PyThreadState;
typedef struct _is PyInterpreterState;

#ifdef __cplusplus
}
#endif
#endif   // !Py_PYTYPEDEFS_H
