/* Frame object interface */

#ifndef Py_CPYTHON_FRAMEOBJECT_H
#  error "this header file must not be included directly"
#endif

/* Standard object interface */

PyAPI_FUNC(PyFrameObject *) PyFrame_New(PyThreadState *, PyCodeObject *,
                                        PyObject *, PyObject *);

/* The rest of the interface is specific for frame objects */

/* Conversions between "fast locals" and locals in dictionary */

PyAPI_FUNC(void) PyFrame_LocalsToFast(PyFrameObject *, int);

/* -- Caveat emptor --
 * The concept of entry frames is an implementation detail of the CPython
 * interpreter. This API is considered unstable and is provided for the
 * convenience of debuggers, profilers and state-inspecting tools. Notice that
 * this API can be changed in future minor versions if the underlying frame
 * mechanism change or the concept of an 'entry frame' or its semantics becomes
 * obsolete or outdated. */

PyAPI_FUNC(int) _PyFrame_IsEntryFrame(PyFrameObject *frame);

PyAPI_FUNC(int) PyFrame_FastToLocalsWithError(PyFrameObject *f);
PyAPI_FUNC(void) PyFrame_FastToLocals(PyFrameObject *);
